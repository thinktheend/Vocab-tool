import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ message: 'Method Not Allowed' });
  
  try {
    const { toolType } = req.body;
    let systemPrompt, userPrompt;

    if (toolType === 'vocabulary') {
        systemPrompt = "You are an AI assistant that ONLY outputs clean, well-formatted Markdown tables for Spanish vocabulary. Your entire response must start with the first table and end with the last table. Do not include any text before or after.";
        userPrompt = buildVocabPrompt(req.body);
    } else if (toolType === 'conversation') {
        systemPrompt = "You are an expert in creating natural, level-appropriate Spanish conversations for language learners. Generate ONLY the conversation content as requested, formatted in Markdown. Do not include any introductory, concluding, or explanatory text whatsoever.";
        userPrompt = buildConvoPrompt(req.body);
    } else {
        return res.status(400).json({ message: 'Invalid tool type specified.' });
    }

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [ { role: "system", content: systemPrompt }, { role: "user", content: userPrompt } ],
      temperature: 0.7,
    });
    
    res.status(200).json({ result: response.choices[0].message.content });
  } catch (error) {
    console.error("API Error:", error);
    res.status(500).json({ message: "An API error occurred.", error: error.message });
  }
}

function buildVocabPrompt({ topic, level, totalsMode, minTotal, maxTotal }) {
  const levelMap = { "1":"Survival", "2":"Beginner", "3":"Intermediate", "4":"Conversational", "5":"Fluent" };
  const presets = { "1":"35–55", "2":"60-85", "3":"90-120", "4":"120-180", "5":"160-240" };
  const lvlTxt = `Level ${level} - ${levelMap[String(level)] || 'Intermediate'}`;
  let totalsLine = (totalsMode === 'custom')
    ? `Total Nouns+Verbs+Descriptive MUST be ${minTotal}–${maxTotal}.`
    : `Total Nouns+Verbs+Descriptive MUST be ${presets[String(level)] || "120-180"}.`;

  return `Create a vocabulary list in Markdown for topic "${topic}" at level "${lvlTxt}".
RULES:
1. ${totalsLine}
2. Sections: Nouns, Verbs, Descriptive Words, Common Phrases, Common Questions.
3. Each section is a Markdown table with the header: | English | Español |
4. Start each table with a header row, e.g., | **Nouns** | **Sustantivos** |
5. For NOUNS/VERBS, use full example sentences. Example: "Here is the **word**."
6. Bold ONLY the target vocabulary word in each sentence using Markdown double asterisks (**word**). This is mandatory.`;
}

function buildConvoPrompt({ topic, level, vocab, numConvos, numSpeakers, turns, tone }) {
  const levelMap = { "1":"Survival (Simple Present)", "2":"Beginner (Core Verbs)", "3":"Intermediate (Past Tenses)", "4": "Conversational (Future/Conditional)", "5": "Fluent (Subjunctive/Commands)" };
  
  let p = `Topic: ${topic}\n`;
  p += `Grammar Level: ${levelMap[level]}\n`;
  
  if (vocab) {
      p += `Vocabulary: Use ONLY the words from this list (plus basic Spanish glue words):\n${vocab}\n`;
  } else {
      p += `Vocabulary: Generate a level-appropriate list based on the topic and grammar level.\n`;
  }
  
  p += `\nCreate the following content formatted in Markdown:\n`;
  p += `— Conversations —\n`;
  p += `• Quantity: ${numConvos}\n`;
  p += `• Speakers: ${numSpeakers}\n`;
  p += `• Turns per conversation (approx): ${turns}\n`;
  p += `• Tone: ${tone}\n`;
  p += `• Formatting: For each line of dialogue, list the speaker (e.g., "Maria:") followed by the text. Immediately below each Spanish line, include the English translation labeled with "(EN)".`;
  
  return p;
}
