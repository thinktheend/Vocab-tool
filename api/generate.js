import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ message: 'Method Not Allowed' });
  
  try {
    const { toolType } = req.body;
    let systemPrompt, userPrompt;

    if (toolType === 'vocabulary') {
        systemPrompt = "You are an AI assistant that ONLY outputs a single, clean, well-formatted Markdown table for Spanish vocabulary. Your entire response must be ONLY the Markdown code for the tables. Do not include any text before or after.";
        userPrompt = buildVocabPrompt(req.body);
    } else if (toolType === 'conversation') {
        systemPrompt = "You are an expert in creating natural, level-appropriate Spanish conversations for language learners. Generate ONLY the conversation content as requested, formatted in Markdown. Do not include any introductory or concluding text.";
        userPrompt = buildConvoPrompt(req.body);
    } else {
        return res.status(400).json({ message: 'Invalid tool type specified.' });
    }

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt }
      ],
      temperature: 0.7,
    });
    
    res.status(200).json({ result: response.choices[0].message.content });
  } catch (error) {
    console.error("Error calling OpenAI API:", error);
    res.status(500).json({ message: "An API error occurred.", error: error.message });
  }
}

// --- PROMPT BUILDER FOR VOCABULARY GENERATOR ---
function buildVocabPrompt({ topic, level, totalsMode, minTotal, maxTotal }) {
  const levelMap = { "1":"L1", "2":"L2", "3":"L3", "4":"L4", "5":"L5" };
  const presets = { "1":"35–55", "2":"60-85", "3":"90-120", "4":"120-180", "5":"160-240" };
  const lvlTxt = levelMap[String(level)] || `Level ${level}`;
  let totalsLine = (totalsMode === 'custom')
    ? `Total Nouns+Verbs+Descriptive MUST be ${minTotal}–${maxTotal}.`
    : `Total Nouns+Verbs+Descriptive MUST be ${presets[String(level)] || "120-180"}.`;

  return `Create a vocabulary list in Markdown for topic "${topic}" at level "${lvlTxt}".
RULES:
1. ${totalsLine}
2. Sections: Nouns, Verbs, Descriptive Words, Phrases, Questions.
3. Each section is a table: | English | Español |
4. Use a header row: | **Nouns** | **Sustantivos** |
5. NOUNS: "Here is the **word**." | "Aquí está el/la **palabra**."
6. VERBS: "I am going to **verb**." | "Yo voy a **verbo**."
7. Bold ONLY the target word with Markdown asterisks (**word**).`;
}

// --- PROMPT BUILDER FOR CONVERSATION GENERATOR ---
function buildConvoPrompt({ topic, level, vocab, numConvos, numSpeakers, turns, tone }) {
  const levelMap = { "1":"L1 - Simple present only.", "2":"L2 - Present, basic verbs.", "3":"L3 - Present, Preterite, Imperfect.", "4":"L4 - Future, Conditional, Perfect.", "5":"L5 - All tenses, Subjunctive, Imperative." };
  
  let p = `Topic: ${topic}\n`;
  p += `Grammar Level: ${levelMap[level] || levelMap["3"]}\n`;
  
  if (vocab) {
      p += `Vocabulary: Use ONLY the words from this list (plus common glue words):\n${vocab}\n`;
  } else {
      p += `Vocabulary: Auto-generate a level-appropriate list based on the topic and level.\n`;
  }

  p += `\nCreate the following content:\n`;
  p += `— Conversations\n`;
  p += `• Quantity: ${numConvos}\n`;
  p += `• Speakers: ${numSpeakers}\n`;
  p += `• Turns per conversation: ~${turns}\n`;
  p += `• Tone: ${tone}\n`;
  p += `• Formatting: For each line of dialogue, list the speaker (e.g., "Amigo 1:") followed by the text. Include English translations below each Spanish line, labeled with "(EN)".`;
  
  return p;
}
