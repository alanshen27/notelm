/** Friendly names for the co-writer checkpoint pickers. */

const TIMESTAMP_PT = /^\d{8}-\d{6}\.pt$/i;

/** Checkpoint filenames from 2026-08-29 on. Product names (Prelude, …) stay on the site. */
export const CHECKPOINT_CODENAMES = [
  "invention",
  "etude",
  "prelude",
  "chaconne",
  "canon",
  "sinfonia",
];

/** Hidden on the public picker. */
export const EXPERIMENTAL_CODENAMES = ["invention", "etude"];

const CODENAME_SET = new Set(CHECKPOINT_CODENAMES);
const EXPERIMENTAL_SET = new Set(EXPERIMENTAL_CODENAMES);

function stemOf(name) {
  return String(name || "").replace(/\.pt$/i, "").toLowerCase();
}

export function isNamedCheckpoint(name) {
  const n = String(name || "").toLowerCase();
  if (!n.endsWith(".pt") || TIMESTAMP_PT.test(n)) return false;
  return true;
}

export function isCodeNameCheckpoint(name) {
  return CODENAME_SET.has(stemOf(name));
}

export function isExperimentalCheckpoint(name) {
  return EXPERIMENTAL_SET.has(stemOf(name));
}

export function checkpointLabel(c) {
  const stem = String(c.name || "").replace(/\.pt$/i, "");
  if (isExperimentalCheckpoint(c.name)) {
    return c.tokenizer ? `Prelude · ${c.tokenizer}` : "Prelude";
  }
  if (stem === "weights") return "weights (default)";
  if (isNamedCheckpoint(c.name)) return stem;
  const tag = [c.model, c.tokenizer].filter(Boolean).join("/");
  const tail = [c.parent, c.name].filter(Boolean).join("/");
  return tag ? `${tag} · ${tail}` : tail || stem;
}

export function cowriterCheckpoints(list) {
  const coded = (list || []).filter(
    (c) => isCodeNameCheckpoint(c.name) && !isExperimentalCheckpoint(c.name)
  );
  return [...coded].sort((a, b) =>
    checkpointLabel(a).localeCompare(checkpointLabel(b), undefined, {
      numeric: true,
    })
  );
}

export function preferCheckpoint(list) {
  const publicList = cowriterCheckpoints(list);
  const hit =
    publicList.find((c) => c.name === "prelude.pt") || publicList[0];
  return hit?.path ?? "";
}
