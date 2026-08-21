/**
 * A stand-in for @deepseek-ai/dsh-tools, so the plugin can be tested without the
 * harness installed. It enforces the parts of the real contract that the plugin must
 * respect — and that we have actually been bitten by:
 *   · `output` is mandatory
 *   · an optional parameter must NOT carry `required: false` (the real compiler throws
 *     "required must be true when present"; a plugin that trips this is skipped, and a
 *     silently skipped plugin looks exactly like a working one)
 *   · ASCII tool names
 */
export function defineTool(def) {
  if (!def || typeof def.name !== "string" || !def.name) throw new Error("tool needs a name");
  if (!/^[a-z0-9_]+$/i.test(def.name)) throw new Error(`tool name must be ASCII: ${def.name}`);
  if (typeof def.description !== "string" || def.description.length < 20) {
    throw new Error(`${def.name}: description must tell the model when to call it`);
  }
  if (!def.output || typeof def.output.render !== "function") {
    throw new Error(`${def.name}: output declaration is mandatory`);
  }
  for (const [k, v] of Object.entries(def.parameters || {})) {
    if ("required" in v && v.required !== true) {
      throw new Error(`${def.name}.${k}: required must be true when present`);
    }
    if (typeof v.description !== "string") throw new Error(`${def.name}.${k}: needs a description`);
  }
  if (typeof def.execute !== "function") throw new Error(`${def.name}: needs execute()`);
  return def;
}
