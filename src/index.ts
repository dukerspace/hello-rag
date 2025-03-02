import WebScraperRAG from "./rag.ts";

async function main(): Promise<void> {
    const rag = new WebScraperRAG();
    await rag.initialize();
  
    // Example: Scrape a website
    const url = 'https://dukerspace.com';
    console.log('Scraping website...');
    await rag.scrapeWebsite(url);
  }

  main().catch(console.error);