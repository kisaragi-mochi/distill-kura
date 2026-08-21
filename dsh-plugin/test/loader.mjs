/** Resolves the harness package to the stub above, so tests need no harness install. */
const STUB = new URL("./stub-tools.mjs", import.meta.url).href;
export function resolve(specifier, context, next) {
  if (specifier === "@deepseek-ai/dsh-tools") return { url: STUB, shortCircuit: true };
  return next(specifier, context);
}
