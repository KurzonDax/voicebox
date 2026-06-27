/**
 * Sentence-aware text chunking for long-form narration.
 *
 * Splits raw text into chunks that respect paragraph and sentence boundaries,
 * keeping each chunk within configurable size limits. Oversized sentences
 * (exceeding ``maxChunkSize``) are kept intact so the UI can flag them for
 * manual editing rather than silently truncating mid-word.
 */

export interface TextChunk {
  id: string;
  text: string;
  charCount: number;
  wordCount: number;
}

/** Normalise line endings and trim surrounding whitespace. */
function normalizeText(text: string): string {
  return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim();
}

/**
 * Split a single paragraph into sentences using terminal punctuation
 * (``.``, ``!``, ``?``) as boundaries. Trailing quotes/brackets stay attached
 * to the preceding sentence. A paragraph with no terminal punctuation is
 * returned as a single element.
 */
function splitParagraphIntoSentences(paragraph: string): string[] {
  const trimmed = paragraph.trim();
  if (!trimmed) {
    return [];
  }

  const matches = trimmed.match(/[^.!?]+[.!?]+(?:["')\]]+)?|[^.!?]+$/g);
  if (!matches || matches.length === 0) {
    return [trimmed];
  }

  return matches.map((sentence) => sentence.trim()).filter(Boolean);
}

/**
 * Chunk ``rawText`` into ``TextChunk`` objects.
 *
 * @param rawText - The full input text.
 * @param targetChunkSize - Soft target — chunks are filled up to this size
 *   before a new chunk starts.
 * @param maxChunkSize - Hard maximum — any single sentence exceeding this is
 *   emitted as its own oversized chunk so the UI can warn the user.
 * @returns Array of chunks with stable ``id``, ``text``, ``charCount``, and
 *   ``wordCount`` fields.
 */
export function chunkText(
  rawText: string,
  targetChunkSize: number,
  maxChunkSize: number,
): TextChunk[] {
  const text = normalizeText(rawText);
  if (!text) {
    return [];
  }

  const safeTarget = Math.max(200, Math.min(targetChunkSize, maxChunkSize));
  const paragraphs = text
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);

  const chunks: string[] = [];
  let current = '';

  const pushCurrent = () => {
    const normalized = current.trim();
    if (!normalized) {
      return;
    }
    chunks.push(normalized);
    current = '';
  };

  for (const paragraph of paragraphs) {
    const sentences = splitParagraphIntoSentences(paragraph);

    for (const sentence of sentences) {
      // Keep sentence integrity. If one sentence exceeds maxChunkSize,
      // keep it as a single oversized chunk and let UI ask for manual edit.
      if (sentence.length > maxChunkSize) {
        pushCurrent();
        chunks.push(sentence);
        continue;
      }

      if (!current) {
        current = sentence;
        continue;
      }

      const candidate = `${current} ${sentence}`;
      if (candidate.length <= safeTarget) {
        current = candidate;
        continue;
      }

      if (candidate.length <= maxChunkSize && current.length < Math.floor(safeTarget * 0.75)) {
        current = candidate;
        continue;
      }

      pushCurrent();
      current = sentence;
    }

    if (current.length >= Math.floor(safeTarget * 0.8)) {
      pushCurrent();
    }
  }

  pushCurrent();

  return chunks.map((chunkTextValue, index) => ({
    id: `chunk-${index + 1}`,
    text: chunkTextValue,
    charCount: chunkTextValue.length,
    wordCount: chunkTextValue.split(/\s+/).filter(Boolean).length,
  }));
}
