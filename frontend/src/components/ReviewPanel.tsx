import { useState } from 'react';
import type { ReviewPayload, ReviewRequest } from '../types';

/** The human gate (`review_draft`): approve, send back with a note, or rewrite. */
export function ReviewPanel({
  payload,
  busy,
  onSubmit,
}: {
  payload: ReviewPayload;
  busy: boolean;
  onSubmit: (req: ReviewRequest) => void;
}) {
  const [mode, setMode] = useState<'choose' | 'reject' | 'edit'>('choose');
  const [note, setNote] = useState('');
  const [text, setText] = useState(payload.draft);

  return (
    <section className="review" aria-label="Review the draft">
      <h3>This draft needs your approval</h3>
      <p className="review-lede">Nothing is added to the conversation until you decide.</p>
      {payload.critique && (
        <p className="critique-note">
          <strong>Critic:</strong> {payload.critique}
        </p>
      )}

      {mode === 'choose' && (
        <div className="review-actions">
          <button className="btn btn-primary" disabled={busy} onClick={() => onSubmit({ decision: 'approve' })}>
            Approve draft
          </button>
          <button className="btn" disabled={busy} onClick={() => setMode('reject')}>
            Send back
          </button>
          <button className="btn" disabled={busy} onClick={() => setMode('edit')}>
            Edit it myself
          </button>
        </div>
      )}

      {mode === 'reject' && (
        <form
          className="review-form"
          onSubmit={(e) => {
            e.preventDefault();
            onSubmit({ decision: 'reject', note });
          }}
        >
          <label htmlFor="review-note">What should change?</label>
          <textarea id="review-note" rows={3} value={note} onChange={(e) => setNote(e.target.value)} placeholder="e.g. Compare the two frameworks on cost" />
          <div className="review-actions">
            <button type="submit" className="btn btn-primary" disabled={busy}>
              Send back for revision
            </button>
            <button type="button" className="btn btn-quiet" onClick={() => setMode('choose')}>
              Cancel
            </button>
          </div>
        </form>
      )}

      {mode === 'edit' && (
        <form
          className="review-form"
          onSubmit={(e) => {
            e.preventDefault();
            onSubmit({ decision: 'edit', text, note });
          }}
        >
          <label htmlFor="review-edit">Your version of the answer</label>
          <textarea id="review-edit" rows={10} value={text} onChange={(e) => setText(e.target.value)} />
          <label htmlFor="review-edit-note">Why you changed it (optional)</label>
          <input id="review-edit-note" value={note} onChange={(e) => setNote(e.target.value)} />
          <div className="review-actions">
            <button type="submit" className="btn btn-primary" disabled={busy || !text.trim()}>
              Use my version
            </button>
            <button type="button" className="btn btn-quiet" onClick={() => setMode('choose')}>
              Cancel
            </button>
          </div>
        </form>
      )}
    </section>
  );
}
