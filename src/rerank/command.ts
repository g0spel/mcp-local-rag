// Parsing of the configured rerank command into an argv vector.

/** An executable and the arguments configured alongside it. */
export interface RerankCommand {
  /** Program to spawn. Must name a directly executable file, not a shim. */
  executable: string
  /** Arguments configured by the operator, before the server's own flags. */
  args: string[]
}

/**
 * Splits the configured command on whitespace. Quoting and other shell syntax
 * are deliberately not interpreted: the command is spawned without a shell, so
 * a quote this parser stripped would not mean what the operator expects.
 *
 * Returns undefined when the string carries no token.
 */
export function parseRerankCommand(command: string): RerankCommand | undefined {
  const [executable, ...args] = command.split(/\s+/).filter((token) => token.length > 0)
  return executable === undefined ? undefined : { executable, args }
}

/**
 * Appends the server's flags after the configured arguments. Each value is its
 * own element, so query text is never read as command syntax.
 */
export function buildRerankArgv(command: RerankCommand, query: string, top: number): string[] {
  return [...command.args, '--query', query, '--top', String(top)]
}
