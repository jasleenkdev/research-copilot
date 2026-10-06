import type { Row, RunStatus } from '../state/run';
import { AGENTS } from '../types';

export const AGENT_LABEL: Record<string, string> = {
  supervisor: 'Supervisor',
  researcher: 'Researcher',
  writer: 'Writer',
  critic: 'Critic',
  finish: 'Finish',
};

const lane = (agent: string) => {
  const i = AGENTS.indexOf(agent as (typeof AGENTS)[number]);
  return i === -1 ? AGENTS.length - 1 : i;
};

function DecisionRow({ row }: { row: Extract<Row, { kind: 'decision' }> }) {
  const finish = row.to === 'finish';
  const to = finish ? 0 : lane(row.to);
  return (
    <li className="row row-decision" style={{ ['--to' as string]: to }}>
      <div className="lanes" aria-hidden="true">
        <span className="dot dot-supervisor" style={{ ['--lane' as string]: 0 }} />
        {!finish && <span className="link" />}
        {!finish && <span className={`dot dot-target dot-${row.to}`} style={{ ['--lane' as string]: to }} />}
      </div>
      <div className="row-body">
        <p className="row-title">
          <span className="who who-supervisor">Supervisor</span>{' '}
          {finish ? 'marks the draft ready' : <>sends it to <span className={`who who-${row.to}`}>{AGENT_LABEL[row.to] ?? row.to}</span></>}
        </p>
        {row.rationale && <p className="row-note">{row.rationale}</p>}
        {row.brief && <p className="row-note">Brief: {row.brief}</p>}
        {row.override && (
          <p className="row-note tone-warn">
            Proposed {row.proposed ? AGENT_LABEL[row.proposed] ?? row.proposed : 'nothing usable'}; overridden: {row.override}
          </p>
        )}
      </div>
    </li>
  );
}

function WorkRow({ row }: { row: Extract<Row, { kind: 'work' }> }) {
  const status = row.running
    ? 'working'
    : row.agent === 'writer'
      ? `wrote draft ${(row.draftIndex ?? 0) + 1}`
      : row.agent === 'critic'
        ? row.verdict === 'approve'
          ? 'approved the draft'
          : row.verdict
            ? `${row.verdict === 'reject' ? 'rejected' : 'could not finish reviewing'} the draft`
            : 'finished'
        : 'wrote notes';
  return (
    <li className={`row row-work${row.running ? ' is-running' : ''}`}>
      <div className="lanes" aria-hidden="true">
        <span className={`dot dot-work dot-${row.agent}`} style={{ ['--lane' as string]: lane(row.agent) }} />
      </div>
      <div className="row-body">
        <p className="row-title">
          <span className={`who who-${row.agent}`}>{AGENT_LABEL[row.agent]}</span>{' '}
          <span className={row.verdict === 'reject' ? 'tone-bad' : row.verdict === 'approve' ? 'tone-good' : undefined}>{status}</span>
        </p>
        {row.critique && <p className="row-note">{row.critique}</p>}
        {row.activity.length > 0 && (
          <ul className="activity">
            {row.activity.map((a, i) => (
              <li key={i} className={a.tone === 'warn' ? 'tone-warn' : undefined}>
                {a.label}
                {a.detail && <> <q>{a.detail}</q></>}
              </li>
            ))}
          </ul>
        )}
      </div>
    </li>
  );
}

export function RouteMap({ rows, status }: { rows: Row[]; status: RunStatus }) {
  return (
    <section className="route" aria-label="How the agents handed off the work">
      <div className="legend">
        {AGENTS.map((a) => (
          <span key={a} className={`legend-item who-${a}`}>
            <span className={`swatch dot-${a}`} aria-hidden="true" />
            {AGENT_LABEL[a]}
          </span>
        ))}
      </div>
      {rows.length === 0 ? (
        <p className="route-empty">
          {status === 'running'
            ? 'Starting the run…'
            : 'Ask a question and each hand-off appears here: who the Supervisor picks, what they search, and what the Critic checks.'}
        </p>
      ) : (
        <ol className="rows" aria-live="polite">
          {rows.map((row) =>
            row.kind === 'decision' ? (
              <DecisionRow key={row.id} row={row} />
            ) : row.kind === 'work' ? (
              <WorkRow key={row.id} row={row} />
            ) : (
              <li key={row.id} className={`row row-marker${row.tone ? ` tone-${row.tone}` : ''}`}>
                <span>{row.label}</span>
              </li>
            ),
          )}
        </ol>
      )}
    </section>
  );
}
