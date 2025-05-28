/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
/* eslint-disable @typescript-eslint/no-unsafe-argument */
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";

export default function CodeBlock({ className, children }: any) {
  const codeString = String(children).replace(/\n$/, "");

  const match = /language-(\w+)/.exec(className ?? "");
  const language = match?.[1] ?? "python";

  return (
    <SyntaxHighlighter
      style={vscDarkPlus}
      language={language}
      PreTag="pre"
      customStyle={{
        borderRadius: "0.5rem",
        padding: "1.1rem",
        fontSize: "1rem",
      }}
    >
      {codeString}
    </SyntaxHighlighter>
  );
}
