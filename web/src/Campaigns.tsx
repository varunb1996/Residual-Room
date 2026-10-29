import { useCallback, useEffect, useState } from "react";
import { api, errorText, type PieceSummary } from "./api";
import type { components } from "./schema";

type Campaign = components["schemas"]["CampaignOut"];
type Preview = components["schemas"]["PreviewOut"];
type Segment = "transitioning" | "practitioner";
type Channel = "email" | "telegram" | "whatsapp";

const SEGMENTS: { id: Segment; label: string }[] = [
  { id: "practitioner", label: "Practitioners" },
  { id: "transitioning", label: "Engineers moving into AI" },
];
const CHANNELS: { id: Channel; label: string }[] = [
  { id: "email", label: "Email" },
  { id: "telegram", label: "Telegram" },
  { id: "whatsapp", label: "WhatsApp (mock)" },
];
const CAMPAIGN_STATUS: Record<string, string> = {
  scheduled: "Scheduled", sending: "Sending", sent: "Sent", failed: "Failed", cancelled: "Cancelled",
};

export default function Campaigns() {
  const [pieces, setPieces] = useState<PieceSummary[]>([]);
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [pieceId, setPieceId] = useState<number | null>(null);
  const [segments, setSegments] = useState<Segment[]>(["practitioner"]);
  const [channels, setChannels] = useState<Channel[]>(["email", "telegram"]);
  const [when, setWhen] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewTab, setPreviewTab] = useState<Channel>("email");
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);

  const refresh = useCallback(async () => {
    const [p, c] = await Promise.all([api.GET("/api/pieces"), api.GET("/api/campaigns")]);
    if (p.data) setPieces(p.data);
    if (c.data) setCampaigns(c.data);
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000); // deliveries and clicks update as they happen
    return () => clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    setPreview(null);
    if (pieceId === null) return;
    api.GET("/api/pieces/{piece_id}/preview", { params: { path: { piece_id: pieceId } } })
      .then(({ data }) => data && setPreview(data));
  }, [pieceId]);

  const approved = pieces.filter((p) => p.state === "approved");
  const toggle = <T,>(list: T[], item: T) => (list.includes(item) ? list.filter((x) => x !== item) : [...list, item]);

  async function schedule() {
    if (pieceId === null) return;
    setMessage(null);
    const { error } = await api.POST("/api/campaigns", {
      body: { piece_id: pieceId, segments, channels, scheduled_at: when ? new Date(when).toISOString() : null },
    });
    if (error) setMessage({ kind: "error", text: errorText(error) });
    else {
      setMessage({ kind: "ok", text: when ? "Scheduled. It appears in the log below." : "Sending now. Watch the log below." });
      setPieceId(null);
    }
    refresh();
  }

  async function cancel(id: number) {
    const { error } = await api.POST("/api/campaigns/{campaign_id}/cancel", { params: { path: { campaign_id: id } } });
    if (error) setMessage({ kind: "error", text: errorText(error) });
    refresh();
  }

  return (
    <>
      <div className="campaign-grid">
        <section className="compose" aria-labelledby="compose-title">
          <h1 id="compose-title">Send a piece</h1>
          <p className="muted small">
            Only approved pieces can go out. Email reaches the test inboxes only, Telegram the test chat only,
            and WhatsApp is a mock that refuses anything outside its rules.
          </p>
          <label className="field">
            Approved piece
            <select value={pieceId ?? ""} onChange={(e) => setPieceId(e.target.value ? Number(e.target.value) : null)}>
              <option value="">{approved.length ? "Choose a piece" : "Nothing approved yet"}</option>
              {approved.map((p) => <option key={p.id} value={p.id}>{p.title}</option>)}
            </select>
          </label>
          <fieldset>
            <legend>Readers</legend>
            {SEGMENTS.map((s) => (
              <label key={s.id} className="toggle">
                <input type="checkbox" checked={segments.includes(s.id)} onChange={() => setSegments(toggle(segments, s.id))} />
                {s.label}
              </label>
            ))}
          </fieldset>
          <fieldset>
            <legend>Channels</legend>
            {CHANNELS.map((c) => (
              <label key={c.id} className="toggle">
                <input type="checkbox" checked={channels.includes(c.id)} onChange={() => setChannels(toggle(channels, c.id))} />
                {c.label}
              </label>
            ))}
          </fieldset>
          <label className="field">
            Send at (leave empty to send now)
            <input type="datetime-local" value={when} onChange={(e) => setWhen(e.target.value)} />
          </label>
          {message && <p className={`notice ${message.kind}`}>{message.text}</p>}
          <div className="row">
            <button className="primary" disabled={pieceId === null || !segments.length || !channels.length} onClick={schedule}>
              {when ? "Schedule" : "Send now"}
            </button>
          </div>
        </section>

        <section className="preview" aria-label="Preview">
          {!preview && (
            <div className="preview-empty">
              <h2>Preview</h2>
              <p className="muted">Choose a piece to see it as each channel will show it.</p>
            </div>
          )}
          {preview && (
            <>
              <div className="preview-tabs" role="tablist">
                {CHANNELS.map((c) => (
                  <button key={c.id} role="tab" aria-selected={previewTab === c.id} onClick={() => setPreviewTab(c.id)}>
                    {c.label}
                  </button>
                ))}
              </div>
              {previewTab === "email" && (
                <>
                  <p className="small"><span className="muted">Subject:</span> {preview.email_subject}</p>
                  {/* Sandboxed: the email HTML is shown, never run. */}
                  <iframe title="Email preview" sandbox="" srcDoc={preview.email_html} />
                </>
              )}
              {previewTab === "telegram" && <pre>{preview.telegram_html}</pre>}
              {previewTab === "whatsapp" && (
                <>
                  <p className="small muted">Sent as an approved template, so only this text goes out.</p>
                  <pre>{preview.whatsapp_text}</pre>
                </>
              )}
            </>
          )}
        </section>
      </div>

      <section className="log">
        <h2>Campaign log</h2>
        {campaigns.length === 0 ? <p className="muted">Nothing sent or scheduled yet.</p> : (
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr><th>Piece</th><th>Readers</th><th>When</th><th>Status</th><th>Deliveries</th><th className="num">Clicks</th><th /></tr>
              </thead>
              <tbody>
                {campaigns.map((c) => (
                  <tr key={c.id}>
                    <td>{c.piece_title}</td>
                    <td>{c.segments.map((s) => SEGMENTS.find((x) => x.id === s)?.label ?? s).join(", ")}</td>
                    <td className="nowrap">{new Date(c.sent_at ?? c.scheduled_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}</td>
                    <td>{CAMPAIGN_STATUS[c.status] ?? c.status}</td>
                    <td><Deliveries campaign={c} /></td>
                    <td className="num" title={c.clicks.map((k) => `${k.channel}, ${k.segment}, ${k.label}: ${k.clicks}`).join("\n")}>
                      {c.clicks.reduce((sum, k) => sum + k.clicks, 0)}
                    </td>
                    <td>{c.status === "scheduled" && <button className="quiet" onClick={() => cancel(c.id)}>Cancel</button>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}

// One line per channel: how many went out, and how many were refused or failed (with the reason on hover).
function Deliveries({ campaign }: { campaign: Campaign }) {
  if (!campaign.deliveries.length) return <span className="muted">None yet</span>;
  return (
    <div className="channel-status">
      {campaign.channels.map((ch) => {
        const ds = campaign.deliveries.filter((d) => d.channel === ch);
        if (!ds.length) return null;
        const sent = ds.filter((d) => d.status === "sent").length;
        const other = ds.filter((d) => d.status !== "sent");
        const label = CHANNELS.find((x) => x.id === ch)?.label.replace(" (mock)", "") ?? ch;
        return (
          <span key={ch} title={other.map((d) => `${d.recipient}: ${d.status}${d.error ? `, ${d.error}` : ""}`).join("\n")}>
            {label}{" "}
            {sent > 0 && <span className="ok-text">{sent} sent</span>}
            {sent > 0 && other.length > 0 && ", "}
            {other.length > 0 && (
              <span className={other.every((d) => d.status === "refused") ? "warn-text" : "bad-text"}>
                {other.length} {other.every((d) => d.status === "refused") ? "refused" : "not delivered"}
              </span>
            )}
          </span>
        );
      })}
    </div>
  );
}
