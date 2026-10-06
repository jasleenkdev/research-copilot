import { citedIds, linkCitations } from './citations';

it('links single and grouped arXiv citations', () => {
  expect(linkCitations('RAGAS [2309.15217] and [2311.09476, 2408.12398v2].')).toBe(
    'RAGAS [2309.15217](https://arxiv.org/abs/2309.15217) and [2311.09476](https://arxiv.org/abs/2311.09476) [2408.12398v2](https://arxiv.org/abs/2408.12398v2).',
  );
});

it('leaves other brackets alone', () => {
  expect(linkCitations('see [1] and [note]')).toBe('see [1] and [note]');
});

it('lists distinct ids in order of first appearance', () => {
  expect(citedIds('[2311.09476] then [2309.15217; 2311.09476]')).toEqual(['2311.09476', '2309.15217']);
});
