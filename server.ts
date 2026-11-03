import express from 'express';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;

// Middleware for parsing JSON with generous limits for large slide texts
app.use(express.json({ limit: '50mb' }));

// Helper to instantiate Gemini client with required User-Agent
function getGenAIClient(customApiKey?: string): GoogleGenAI {
  const key = customApiKey || process.env.GEMINI_API_KEY;
  return new GoogleGenAI({
    apiKey: key,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

const SYSTEM_INSTRUCTION =
  `You are an academic examiner. Generate study materials STRICTLY and SOLELY from the provided slide text. Do not invent, extrapolate, or introduce external facts, concepts, or outside course syllabus material not explicitly written in the slides. If a concept is not in the text, omit it.

CONTENT FILTERING & RELEVANCE RULES:
1. IGNORE ADMINISTRATIVE & INTRODUCTORY SLIDES:
   - Completely disregard cover/title slides, agenda slides, professor/instructor biographies, grading policies, attendance rules, course announcements, office hours, syllabus snapshots, and "Questions?" or "Thank You" ending slides.
   - Do NOT generate any flashcards or test questions based on dates, speaker names, room numbers, course codes, or exam policies.

2. TARGET HIGH-YIELD ACADEMIC SUBSTANCE ONLY:
   - Focus exclusively on core conceptual material: formal definitions, frameworks, formulas, step-by-step methodologies, cause-and-effect relationships, and business/mathematical principles.
   - If a slide contains only a stock image, decorative quote, or generic header with no factual substance, skip it entirely.

3. STRICT GROUNDING:
   - Every question and flashcard must test knowledge directly applicable to the academic subject matter covered in the substantive body slides.`;

// Candidate models in priority order.
// 'gemini-3.1-flash-lite' has higher free tier quotas, followed by 'gemini-flash-latest' and 'gemini-3.8-flash'.
const CANDIDATE_MODELS = [
  'gemini-3.1-flash-lite',
  'gemini-flash-latest',
  'gemini-3.8-flash',
];

// Health check endpoint
app.get('/api/health', (req, res) => {
  const hasKey = Boolean(process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY.trim() !== '');
  res.json({
    status: 'ok',
    configured: hasKey,
    candidateModels: CANDIDATE_MODELS,
  });
});

// Helper to construct structured prompt
function buildPrompt(courseName: string, courseCode: string, weekTitle: string, slideText: string): string {
  return `Generate weekly academic exam preparation materials for the course "${courseName}" (${courseCode}), ${weekTitle}, STRICTLY from the substantive body slide text.

SLIDE CONTENT:
"""
${slideText}
"""

CONTENT FILTERING & RELEVANCE RULES (MANDATORY):
1. IGNORE ADMINISTRATIVE & INTRODUCTORY SLIDES:
   - Disregard cover/title slides, agenda, professor/instructor biographies, grading policies, attendance rules, announcements, office hours, syllabus snapshots, and "Questions?" or "Thank You" ending slides.
   - Do NOT generate any flashcards or test questions based on dates, speaker names, room numbers, course codes, or exam policies.
2. TARGET HIGH-YIELD ACADEMIC SUBSTANCE ONLY:
   - Focus exclusively on core conceptual material: formal definitions, frameworks, formulas, step-by-step methodologies, cause-and-effect relationships, and business/mathematical principles.
   - If a slide contains only a stock image, decorative quote, or generic header with no factual substance, skip it entirely.
3. STRICT GROUNDING:
   - Every question and flashcard must test knowledge directly applicable to the academic subject matter covered in the substantive body slides.

STRICT MANDATORY DELIVERABLES IN VALID JSON:
{
  "summary": {
    "executive_overview": "A rigorous 2-3 paragraph academic synthesis of the core topics covered solely in these slides.",
    "core_takeaways": [
      "Takeaway 1 citing substantive academic principles",
      "Takeaway 2",
      "Takeaway 3",
      "Takeaway 4",
      "Takeaway 5",
      "Takeaway 6",
      "Takeaway 7"
    ],
    "key_formulas_and_frameworks": [
      {
        "name": "Formula or Model Name from Slides",
        "details": "Mathematical formula, variable breakdown, or framework matrices found explicitly on slides.",
        "slide_reference": "Slide X"
      }
    ],
    "key_definitions": [
      {
        "term": "Term directly from slides",
        "definition": "Strict definition given in slides.",
        "slide_reference": "Slide X"
      }
    ]
  },
  "flashcards": [
    {
      "front": "Academic Term, Theoretical Concept, or Quantitative Prompt",
      "back": "Detailed definition, formula, or explanation directly from slides",
      "slide_reference": "Slide X"
    }
  ],
  "practice_questions": [
    // EXACTLY 40 QUESTIONS NUMBERED 1 TO 40:
    // Questions 1 to 20: Multiple Choice Questions (4 options: A, B, C, D)
    {
      "question_number": 1,
      "type": "mcq",
      "question": "Clear academic question testing slide concept",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "B",
      "explanation": "Detailed explanation citing why B is correct strictly from the substantive slide.",
      "slide_reference": "Slide 2"
    },
    // Questions 21 to 30: Rapid Recall / Fill-in-the-Blank targeting exact terminology
    {
      "question_number": 21,
      "type": "fill_in_the_blank",
      "sentence_with_blank": "In this domain, ________ represents the core mechanism described on the slide.",
      "target_term": "Target Keyword",
      "acceptable_variations": ["target keyword", "keyword"],
      "explanation": "Rationale citing the specific slide definition.",
      "slide_reference": "Slide 5"
    },
    // Questions 31 to 40: Short-answer conceptual & analytical application questions
    {
      "question_number": 31,
      "type": "short_answer",
      "question": "Analytical scenario or problem requiring application of slide framework.",
      "model_answer": "Comprehensive exemplar answer showing full marks derivation.",
      "key_grading_points": [
        "Identified primary variable from slide",
        "Applied core formula correctly",
        "Explained academic trade-off"
      ],
      "slide_reference": "Slide 9"
    }
  ]
}

CRITICAL RULES:
1. STRICTLY 40 practice questions total: Q1-20 must be MCQ, Q21-30 must be fill_in_the_blank, Q31-40 must be short_answer.
2. Every single question and flashcard MUST have a valid "slide_reference" citing the substantive slide (e.g. "Slide 3").
3. Completely ignore administrative slides (dates, professor names, emails, room numbers, grading policies, agendas, Q&A/Thank you).
4. Do NOT extrapolate or introduce external facts beyond the provided slides.
5. Output MUST be valid, parseable JSON.`;
}

// Full Server-Side Generation Endpoint with Multi-Model Quota Resilience
app.post('/api/generate-exam-deck', async (req, res) => {
  try {
    const { courseName, courseCode, weekTitle, slideText, apiKeyOverride } = req.body;
    const clientKey = (req.headers['x-gemini-api-key'] as string) || apiKeyOverride;

    if (!slideText || typeof slideText !== 'string' || slideText.trim().length < 20) {
      return res.status(400).json({
        error: 'Slide text is required and must contain at least 20 characters.',
      });
    }

    if (!process.env.GEMINI_API_KEY && !clientKey) {
      return res.status(500).json({
        error: 'GEMINI_API_KEY is not configured on the server. Please check the Settings > Secrets panel.',
      });
    }

    const ai = getGenAIClient(clientKey);
    const prompt = buildPrompt(courseName || 'Course', courseCode || 'COURSE101', weekTitle || 'Week 1', slideText);

    let lastError: any = null;
    let responseText: string | undefined;
    let successfulModel = '';

    // Attempt candidate models to survive single-model free-tier rate limits
    for (const model of CANDIDATE_MODELS) {
      try {
        console.log(`[Examiner] Attempting generation with model: ${model}`);
        const response = await ai.models.generateContent({
          model: model,
          contents: prompt,
          config: {
            systemInstruction: SYSTEM_INSTRUCTION,
            responseMimeType: 'application/json',
          },
        });

        if (response.text) {
          responseText = response.text;
          successfulModel = model;
          console.log(`[Examiner] Generation succeeded with model: ${model}`);
          break;
        }
      } catch (err: any) {
        lastError = err;
        console.warn(`[Examiner] Model ${model} encountered error:`, err?.message || err);

        const isQuotaExceeded =
          err?.status === 'RESOURCE_EXHAUSTED' ||
          err?.status === 429 ||
          err?.message?.includes('429') ||
          err?.message?.includes('RESOURCE_EXHAUSTED') ||
          err?.message?.includes('Quota exceeded');

        if (isQuotaExceeded) {
          console.warn(`[Examiner] Quota limit reached on ${model}. Trying next candidate model...`);
          continue;
        }

        const isNotFound =
          err?.status === 'NOT_FOUND' ||
          err?.status === 404 ||
          err?.message?.includes('404') ||
          err?.message?.includes('NOT_FOUND');

        if (isNotFound) {
          console.warn(`[Examiner] Model ${model} not found or unsupported. Trying next candidate model...`);
          continue;
        }
      }
    }

    if (!responseText) {
      const isQuotaExhausted =
        lastError?.status === 'RESOURCE_EXHAUSTED' ||
        lastError?.status === 429 ||
        lastError?.message?.includes('RESOURCE_EXHAUSTED') ||
        lastError?.message?.includes('Quota exceeded');

      if (isQuotaExhausted) {
        return res.status(429).json({
          error:
            'Free tier request limit exceeded for your project. Upgrading to a paid tier increases your quota, and you can use the Settings > Secrets panel to select a billing-enabled API key.',
          details: lastError?.message,
        });
      }

      throw lastError || new Error('No content received from any candidate Gemini model.');
    }

    // Sanitize and parse JSON
    let cleaned = responseText.trim();
    if (cleaned.startsWith('```json')) {
      cleaned = cleaned.replace(/^```json\s*/, '').replace(/\s*```$/, '');
    } else if (cleaned.startsWith('```')) {
      cleaned = cleaned.replace(/^```\s*/, '').replace(/\s*```$/, '');
    }

    const parsedData = JSON.parse(cleaned);

    res.json({
      success: true,
      modelUsed: successfulModel,
      data: parsedData,
    });
  } catch (err: any) {
    console.error('Server Gemini API error:', err);
    res.status(500).json({
      error: err.message || 'Failed to generate study pack using server-side Gemini API.',
    });
  }
});

// Server-Side Streaming Endpoint (Server-Sent Events) with Quota Fallback
app.post('/api/generate-exam-deck-stream', async (req, res) => {
  try {
    const { courseName, courseCode, weekTitle, slideText, apiKeyOverride } = req.body;
    const clientKey = (req.headers['x-gemini-api-key'] as string) || apiKeyOverride;

    if (!slideText || typeof slideText !== 'string' || slideText.trim().length < 20) {
      return res.status(400).json({
        error: 'Slide text is required.',
      });
    }

    if (!process.env.GEMINI_API_KEY && !clientKey) {
      return res.status(500).json({
        error: 'GEMINI_API_KEY is not configured on the server.',
      });
    }

    const ai = getGenAIClient(clientKey);
    const prompt = buildPrompt(courseName || 'Course', courseCode || 'COURSE101', weekTitle || 'Week 1', slideText);

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    let streamEstablished = false;
    let lastError: any = null;

    for (const model of CANDIDATE_MODELS) {
      try {
        const responseStream = await ai.models.generateContentStream({
          model: model,
          contents: prompt,
          config: {
            systemInstruction: SYSTEM_INSTRUCTION,
            responseMimeType: 'application/json',
          },
        });

        streamEstablished = true;
        res.write(`data: ${JSON.stringify({ status: 'started', model: model })}\n\n`);

        for await (const chunk of responseStream) {
          const chunkText = chunk.text;
          if (chunkText) {
            res.write(`data: ${JSON.stringify({ chunk: chunkText })}\n\n`);
          }
        }

        res.write('data: [DONE]\n\n');
        res.end();
        break;
      } catch (err: any) {
        lastError = err;
        console.warn(`[Stream] Model ${model} failed, trying next...`, err?.message || err);
      }
    }

    if (!streamEstablished) {
      res.write(
        `data: ${JSON.stringify({
          error:
            lastError?.message ||
            'Daily request limit exceeded. Check your API key in Settings > Secrets.',
        })}\n\n`
      );
      res.end();
    }
  } catch (err: any) {
    console.error('Streaming Gemini API error:', err);
    res.write(`data: ${JSON.stringify({ error: err.message || 'Streaming failed' })}\n\n`);
    res.end();
  }
});

// Vite Middleware for Development / Static Hosting for Production
async function setupViteOrStatic() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: false,
        watch: null,
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

setupViteOrStatic();
