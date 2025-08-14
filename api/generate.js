import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method Not Allowed' });
  }
  
  try {
    const { topic, level, totalsMode, minTotal, maxTotal } = req.body;

    const systemPrompt = "You are an AI assistant that ONLY outputs a single, clean, well-formatted Markdown table for Spanish vocabulary. You MUST NOT include any introductory sentences, conversational text, or explanations before or after the table. Your entire response must be ONLY the Markdown code for the tables.";
    const userPrompt = buildUserPrompt(topic, level, totalsMode, minTotal, maxTotal);

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt }
      ],
      temperature: 0.6,
    });
    
    res.status(200).json({ result: response.choices[0].message.content });

  } catch (error) {
    console.error("Error calling OpenAI API:", error);
    res.status(500).json({ message: "Failed to generate vocabulary.", error: error.message });
  }
}

// Helper function to build the detailed prompt for the AI
function buildUserPrompt(topic, level, totalsMode, minTotal, maxTotal) {
  const levelMap = { "1": "Level 1 — Survival", "2": "Level 2 — Beginner", "3": "Level 3 — Intermediate", "4": "Level 4 — Conversational", "5": "Level 5 — Fluent" };
  const presets = { "1": "35–55", "2": "60-85", "3": "90-120", "4": "120-180", "5": "160-240"};
  const lvlTxt = levelMap[String(level)] || `Level ${level}`;

  let totalsLine;
  if (totalsMode === 'custom' && minTotal && maxTotal) {
    totalsLine = `Total Nouns+Verbs+Descriptive words MUST be in the range of ${minTotal}–${maxTotal}.`;
  } else {
    totalsLine = `Total Nouns+Verbs+Descriptive words MUST be in the range of ${presets[String(level)] || "120-180"}.`;
  }

  let p = `Create a vocabulary list in a single Markdown document with multiple tables for the topic "${topic}" at level "${lvlTxt}".\n`;
  p += `RULES:\n`;
  p += `1. ${totalsLine}\n`;
  p += `2. Section order: Nouns (with logical subcategories), Verbs, Descriptive Words, Common Phrases, Common Questions.\n`;
  p += `3. Each section must be its own Markdown table with the header: | English | Español |\n`;
  p += `4. Insert a section header row as the first data row of each table. Example: | **Nouns** | **Sustantivos** |\n`;
  p += `5. NOUNS: Format as full sentences. Example: "Here is the **tent**." | "Aquí está la **tienda de campaña**."\n`;
  p += `6. VERBS: Format as full sentences. Example: "I am going to **pack** the tent." | "Yo voy a **empacar** la tienda de campaña."\n`;
  p += `7. Bold ONLY the target vocabulary word in each sentence using Markdown asterisks (**word**).\n`;
  return p;
}
