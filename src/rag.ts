import { pipeline, PipelineType } from '@xenova/transformers'
import * as cheerio from 'cheerio'
import { ChromaClient, Collection } from 'chromadb'
import { Document } from 'langchain/document'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import puppeteer from 'puppeteer'
import TurndownService from 'turndown'

interface QueryResult {
  question: string
  results: Array<{
    content: string
    metadata: Record<string, any>
    distance: number
  }>
}

interface ProcessingResult {
  chunks_processed: number
  source_url: string
}

class WebScraperRAG {
  private chroma: ChromaClient
  private turndownService: TurndownService
  private textSplitter: RecursiveCharacterTextSplitter
  private collection: Collection | null = null
  private embedder: any = null

  constructor() {
    this.chroma = new ChromaClient()
    this.turndownService = new TurndownService()
    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    })
  }

  async initialize(): Promise<void> {
    // Initialize the embedding model
    this.embedder = await pipeline('feature-extraction' as PipelineType, 'Xenova/all-MiniLM-L6-v2')

    // Create or get collection
    this.collection = await this.chroma.getOrCreateCollection({
      name: 'web_content',
      metadata: { description: 'Scraped web content embeddings' },
    })
  }

  async scrapeWebsite(url: string) {
    try {
      // Launch browser
      const browser = await puppeteer.launch({
        headless: false,
      })

      const page = await browser.newPage()
      await page.goto(url, { waitUntil: 'networkidle0' })

      // Get page content
      const html = await page.content()

      // Parse HTML with cheerio
      const $ = cheerio.load(html)
      const title = $('title').text().trim()
      console.log('Title:', title)

      // Remove unwanted elements
      $('script, style, nav, header,footer, ads').remove()

      // Get main content
      const mainContent = $('body').html() || ''

      // Convert HTML to Markdown
      const markdown = this.turndownService.turndown(mainContent)

      // Find and visit internal links using a queue
      const visited = new Set<string>()
      const queue: string[] = [url] // Start with the given URL

      while (queue.length > 0) {
        const currentUrl = queue.shift() // Dequeue
        if (!currentUrl || visited.has(currentUrl)) continue

        visited.add(currentUrl)
        console.log(`Processing: ${currentUrl}`)

        // Process content
        await this.processContent(markdown, currentUrl)

        // Extract internal links
        await page.goto(currentUrl, { waitUntil: 'networkidle0' })
        const newHtml = await page.content()
        const $$ = cheerio.load(newHtml)

        $$('a[href]').each((_, element) => {
          let link = $$(element).attr('href')
          if (link && link.startsWith('/')) link = new URL(link, url).href // Convert to absolute URL

          if (link && link.startsWith(url) && !visited.has(link)) {
            queue.push(link) // Add to queue
          }
        })
      }

      await browser.close()
    } catch (error) {
      console.error('Error scraping website:', error)
      throw error
    }
  }

  async processContent(content: string, url: string): Promise<ProcessingResult> {
    if (!this.collection) {
      throw new Error('Collection not initialized. Call initialize() first.')
    }

    // Split content into chunks
    const chunks: Document[] = await this.textSplitter.createDocuments([content])
    const textChunks: string[] = chunks.map((chunk) => chunk.pageContent)

    // Generate embeddings for each chunk
    const embeddings: number[][] = await Promise.all(
      textChunks.map(async (chunk) => {
        const output = await this.embedder(chunk, {
          pooling: 'mean',
          normalize: true,
        })
        return Array.from(output.data)
      })
    )

    // Store in ChromaDB
    const documents = textChunks
    const ids = textChunks.map((_, index) => `${url.replace(/[^a-zA-Z0-9]/g, '_')}-${index}`)
    const metadatas = textChunks.map(() => ({ source_url: url }))

    await this.collection.add({
      ids,
      embeddings,
      documents,
      metadatas,
    })

    return {
      chunks_processed: textChunks.length,
      source_url: url,
    }
  }

  async query(question: string, numResults: number = 3): Promise<QueryResult> {
    if (!this.collection) {
      throw new Error('Collection not initialized. Call initialize() first.')
    }

    try {
      // Generate embedding for the question
      const questionEmbedding = await this.embedder(question, {
        pooling: 'mean',
        normalize: true,
      })

      // Query the vector database
      const results = await this.collection.query({
        queryEmbeddings: Array.from(questionEmbedding.data),
        nResults: numResults,
      })

      return {
        question,
        results: results.documents[0].map((doc: string, i: number) => ({
          content: doc,
          metadata: results.metadatas[0][i],
          distance: results.distances[0][i],
        })),
      }
    } catch (error) {
      console.error('Error querying database:', error)
      throw error
    }
  }
}

export default WebScraperRAG
