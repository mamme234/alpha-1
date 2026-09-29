/**
 * Alpha Tools — a safe arithmetic evaluator.
 *
 * Deliberately not `eval` and not `new Function`: tool arguments can originate
 * from model output or retrieved text, so the calculator is a small recursive
 * descent parser over a whitelisted grammar with no property access, no
 * identifiers other than whitelisted functions, and no string handling.
 */

import { AlphaToolError } from "../core/errors";

type Token =
  | { kind: "number"; value: number }
  | { kind: "identifier"; value: string }
  | { kind: "operator"; value: string }
  | { kind: "paren"; value: "(" | ")" }
  | { kind: "comma" };

const FUNCTIONS: Record<string, { arity: number; fn: (...args: number[]) => number }> = {
  sqrt: { arity: 1, fn: Math.sqrt },
  abs: { arity: 1, fn: Math.abs },
  floor: { arity: 1, fn: Math.floor },
  ceil: { arity: 1, fn: Math.ceil },
  round: { arity: 1, fn: Math.round },
  exp: { arity: 1, fn: Math.exp },
  log: { arity: 1, fn: Math.log },
  log10: { arity: 1, fn: Math.log10 },
  sin: { arity: 1, fn: Math.sin },
  cos: { arity: 1, fn: Math.cos },
  tan: { arity: 1, fn: Math.tan },
  min: { arity: -1, fn: (...args) => Math.min(...args) },
  max: { arity: -1, fn: (...args) => Math.max(...args) },
  pow: { arity: 2, fn: Math.pow },
};

function tokenize(input: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < input.length) {
    const ch = input[i];
    if (ch === " " || ch === "\t" || ch === "\n") {
      i++;
      continue;
    }
    if (/[0-9]/.test(ch) || (ch === "." && /[0-9]/.test(input[i + 1] ?? ""))) {
      let literal = "";
      while (i < input.length && /[0-9._]/.test(input[i])) {
        if (input[i] !== "_") literal += input[i];
        i++;
      }
      const value = Number(literal);
      if (!Number.isFinite(value)) {
        throw new AlphaToolError(`"${literal}" is not a valid number`, { expression: input });
      }
      tokens.push({ kind: "number", value });
      continue;
    }
    if (/[a-zA-Z]/.test(ch)) {
      let name = "";
      while (i < input.length && /[a-zA-Z0-9]/.test(input[i])) name += input[i++];
      tokens.push({ kind: "identifier", value: name });
      continue;
    }
    if ("+-*/%^".includes(ch)) {
      tokens.push({ kind: "operator", value: ch });
      i++;
      continue;
    }
    if (ch === "(" || ch === ")") {
      tokens.push({ kind: "paren", value: ch });
      i++;
      continue;
    }
    if (ch === ",") {
      tokens.push({ kind: "comma" });
      i++;
      continue;
    }
    throw new AlphaToolError(`unexpected character "${ch}"`, { expression: input, position: i });
  }
  return tokens;
}

export class ExpressionParser {
  private readonly source: string;
  private tokens: Token[];
  private position = 0;

  constructor(source: string) {
    this.source = source;
    this.tokens = tokenize(source);
  }

  parse(): number {
    if (this.tokens.length === 0) {
      throw new AlphaToolError("expression is empty", { expression: this.source });
    }
    const value = this.parseExpression();
    if (this.position < this.tokens.length) {
      throw new AlphaToolError("unexpected trailing input", {
        expression: this.source,
        at: this.position,
      });
    }
    if (!Number.isFinite(value)) {
      throw new AlphaToolError("expression did not evaluate to a finite number", {
        expression: this.source,
      });
    }
    return value;
  }

  private peek(): Token | undefined {
    return this.tokens[this.position];
  }

  private parseExpression(): number {
    let left = this.parseTerm();
    for (;;) {
      const token = this.peek();
      if (token?.kind === "operator" && (token.value === "+" || token.value === "-")) {
        this.position++;
        const right = this.parseTerm();
        left = token.value === "+" ? left + right : left - right;
        continue;
      }
      return left;
    }
  }

  private parseTerm(): number {
    let left = this.parseUnary();
    for (;;) {
      const token = this.peek();
      if (token?.kind === "operator" && "*%/".includes(token.value)) {
        this.position++;
        const right = this.parseUnary();
        if ((token.value === "/" || token.value === "%") && right === 0) {
          throw new AlphaToolError("division by zero", { expression: this.source });
        }
        if (token.value === "*") left *= right;
        else if (token.value === "/") left /= right;
        else left %= right;
        continue;
      }
      return left;
    }
  }

  private parseUnary(): number {
    const token = this.peek();
    if (token?.kind === "operator" && (token.value === "-" || token.value === "+")) {
      this.position++;
      const value = this.parseUnary();
      return token.value === "-" ? -value : value;
    }
    return this.parsePower();
  }

  private parsePower(): number {
    const base = this.parsePrimary();
    const token = this.peek();
    if (token?.kind === "operator" && token.value === "^") {
      this.position++;
      const exponent = this.parseUnary();
      return base ** exponent;
    }
    return base;
  }

  private parsePrimary(): number {
    const token = this.peek();
    if (!token) throw new AlphaToolError("unexpected end of expression", { expression: this.source });
    if (token.kind === "number") {
      this.position++;
      return token.value;
    }
    if (token.kind === "paren" && token.value === "(") {
      this.position++;
      const value = this.parseExpression();
      const closing = this.peek();
      if (!closing || closing.kind !== "paren" || closing.value !== ")") {
        throw new AlphaToolError("missing closing parenthesis", { expression: this.source });
      }
      this.position++;
      return value;
    }
    if (token.kind === "identifier") {
      const name = token.value;
      const fn = FUNCTIONS[name];
      if (!fn) {
        throw new AlphaToolError(`unknown function "${name}"`, { expression: this.source });
      }
      this.position++;
      const opening = this.peek();
      if (!opening || opening.kind !== "paren" || opening.value !== "(") {
        throw new AlphaToolError(`"${name}" must be called with parentheses`, {
          expression: this.source,
        });
      }
      this.position++;
      const args: number[] = [];
      if (this.peek()?.kind !== "paren") {
        args.push(this.parseExpression());
        while (this.peek()?.kind === "comma") {
          this.position++;
          args.push(this.parseExpression());
        }
      }
      const closing = this.peek();
      if (!closing || closing.kind !== "paren" || closing.value !== ")") {
        throw new AlphaToolError(`missing closing parenthesis for "${name}"`, {
          expression: this.source,
        });
      }
      this.position++;
      if (fn.arity >= 0 && args.length !== fn.arity) {
        throw new AlphaToolError(`"${name}" expects ${fn.arity} argument(s)`, {
          expression: this.source,
        });
      }
      if (fn.arity === -1 && args.length === 0) {
        throw new AlphaToolError(`"${name}" expects at least one argument`, {
          expression: this.source,
        });
      }
      return fn.fn(...args);
    }
    throw new AlphaToolError("unexpected token in expression", {
      expression: this.source,
      token: token.kind === "operator" ? token.value : token.kind,
    });
  }
}

/** Evaluate an arithmetic expression. Throws `AlphaToolError` on bad input. */
export function evaluateExpression(expression: string): number {
  if (expression.length > 500) {
    throw new AlphaToolError("expression is too long", { length: expression.length });
  }
  return new ExpressionParser(expression).parse();
}
