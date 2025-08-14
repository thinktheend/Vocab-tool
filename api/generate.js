// Import the OpenAI library
import OpenAI from 'openai';

// Create an OpenAI client with your API key from environment variables
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// The main serverless function
export default async function handler(req, res) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method Not Allowed' });
  }
  
  try {
    const { topic, level } = req.body;

    // Use the function from your original file to build the prompt.
    // NOTE: This logic needs to be on the server now.
    const prompt = buildFullPrompt(topic, level);

    // Make the API call to OpenAI
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo", // You can use "gpt-4" if you have access and prefer it
      messages: [
        {
          role: "system",
          content: "You are a helpful assistant designed to generate Spanish vocabulary lists based on strict instructions."
        },
        {
          role: "user",
          content: prompt
        }
      ],
      temperature: 0.7,
    });
    
    // Send the AI's response back to the frontend
    res.status(200).json({ result: response.choices[0].message.content });

  } catch (error) {
    console.error("Error calling OpenAI API:", error);
    res.status(500).json({ message: "Failed to generate vocabulary.", error: error.message });
  }
}


// --- HELPER FUNCTION TO BUILD THE PROMPT ---
// I have copied the core logic from your previous HTML file here.
function buildFullPrompt(topic, level) {
  const levelMap = {
      "1": "Level 1 — Survival",
      "2": "Level 2 — Beginner",
      "3": "Level 3 — Intermediate",
      "4": "Level 4 — Conversational",
      "5": "Level 5 — Fluent"
  };
  const lvlTxt = levelMap[String(level)] || ("Level " + level);
  const totals = { "1": "35–55", "2": "60-85", "3": "90-120", "4": "120-180", "5": "160-240"};
  const totalsLine = "Stay within " + lvlTxt + " ranges; total words: " + (totals[String(level)] || "120-180") + " (Nouns+Verbs+Descriptive only).";

  let p = "Output Mode: ChatGPT Inline\n\n";
  p += `You are creating a VOCABULARY LIST in MARKDOWN TABLES for the topic: ${topic} and level: ${lvlTxt}.\n`;
  p += "Follow these rules strictly:\n\n";
  p += "- Produce the FULL list for every section—no omissions, no ellipses.\n";
  p += "- Use ONLY markdown tables.\n";
  p += "- Section order: Nouns (with subcategories), Verbs, Descriptive Words, Common Questions, Common Phrases.\n";
  p += "- **Nouns must be full sentences**: English: \"Here is/are the [noun].\" · Spanish: \"Aquí está(n) el/la/los/las [sustantivo].\"\n";
  p += "- **Verbs must be full sentences**: English: \"I am going to **[verb]** the [noun].\" · Spanish: \"Yo voy a **[verbo]** el/la [sustantivo].\"\n";
  p += `- ${totalsLine}\n\n`;
  p += `Page Title (as plain text, not a heading): ${topic} — Full Vocabulary (${lvlTxt})\n`;
  return p;
}
