// arXiv ids as the Writer cites them: [2309.15217] or [2309.15217, 2311.09476].
const ID = String.raw`\d{4}\.\d{4,5}(?:v\d+)?`;
const GROUP = new RegExp(String.raw`\[(${ID}(?:\s*[,;]\s*${ID})*)\]`, 'g');

export const arxivUrl = (id: string) => `https://arxiv.org/abs/${id}`;

/** Turn bracketed arXiv ids into markdown links, so the renderer can draw them as chips. */
export function linkCitations(markdown: string): string {
  return markdown.replace(GROUP, (_m, ids: string) =>
    ids
      .split(/\s*[,;]\s*/)
      .map((id) => `[${id}](${arxivUrl(id)})`)
      .join(' '),
  );
}

/** Every distinct arXiv id cited in the text, in order of first appearance. */
export function citedIds(markdown: string): string[] {
  const seen = new Set<string>();
  for (const m of markdown.matchAll(GROUP)) for (const id of m[1].split(/\s*[,;]\s*/)) seen.add(id);
  return [...seen];
}
