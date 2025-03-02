import ollama from 'ollama'
import WebScraperRAG from './rag.ts'

async function generateAnswer(question) {
  const rag = new WebScraperRAG()
  await rag.initialize()

  const queryResult = await rag.query(cleanContent(question))

  const prompt = `ข้อมูลที่เกี่ยวข้อง: "${
    queryResult.results.sort((a, b) => b.distance - a.distance)?.[0]?.content
  }" คำถาม: "${question}"`

  console.log('prompt :', prompt)

  const response = await ollama.chat({
    model: 'phi4',
    messages: [{ role: 'user', content: prompt }],
  })

  const content = response.message.content
  console.log('🟢 คำตอบ:', content)
}

function cleanContent(content: string) {
  return content
    .replace(/\* \* \*/g, '') // Remove asterisks dividers
    .replace(/[-]{2,}/g, '') // Remove excessive dashes
    .replace(/\n\s*\n/g, '\n') // Remove extra empty lines
    .trim() // Trim extra spaces at start and end
}

generateAnswer('ทำงาน BANPU Public Company Limited. มากี่ปี ปีไหน ถึงปีไหน?')
