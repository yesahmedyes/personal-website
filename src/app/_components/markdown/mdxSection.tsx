import { MDXProvider } from "@mdx-js/react";
import Derivation from "./derivation";
import Figure from "./figure";
import Lemma from "./lemma";
import React from "react";
import Info from "./info";
import CodeBlock from "./codeBlock";
import Detour from "./detour";
import Algorithm from "./algorithm";

interface MDXSectionProps {
  title?: string;
  heading?: string;
  children: React.ReactNode;
}

const components = {
  h1: (props: React.HTMLAttributes<HTMLHeadingElement>) => <h1 {...props} className="text-3xl font-semibold font-serif pb-5" />,
  h2: (props: React.HTMLAttributes<HTMLHeadingElement>) => <h2 {...props} className="text-2xl font-semibold font-serif pb-3" />,
  h3: (props: React.HTMLAttributes<HTMLHeadingElement>) => <h3 {...props} className="text-xl font-semibold font-serif pb-2" />,
  h4: (props: React.HTMLAttributes<HTMLHeadingElement>) => <h4 {...props} className="text-lg font-semibold font-serif pb-2" />,
  p: (props: React.HTMLAttributes<HTMLParagraphElement>) => <p {...props} className="leading-loose" />,
  ul: (props: React.HTMLAttributes<HTMLUListElement>) => <ul {...props} className="" />,
  ol: (props: React.HTMLAttributes<HTMLOListElement>) => <ol {...props} className="" />,
  li: (props: React.HTMLAttributes<HTMLLIElement>) => <li {...props} className="" />,
  blockquote: (props: React.HTMLAttributes<HTMLQuoteElement>) => <blockquote {...props} className="" />,
  code: CodeBlock,
  pre: (props: React.HTMLAttributes<HTMLPreElement>) => <pre {...props} className="" />,
  strong: (props: React.HTMLAttributes<HTMLElement>) => <span {...props} className="leading-normal font-semibold" />,
  em: (props: React.HTMLAttributes<HTMLElement>) => <em {...props} className="" />,
  a: (props: React.AnchorHTMLAttributes<HTMLAnchorElement>) => <a {...props} className="text-blue-500 hover:text-blue-600 hover:underline" />,
  br: (props: React.HTMLAttributes<HTMLBRElement>) => <br {...props} className="" />,
  Margin: (props: React.HTMLAttributes<HTMLDivElement>) => <div {...props} className="my-4" />,
  Figure,
  Derivation,
  Detour,
  Info,
  Lemma,
  Algorithm,
};

export default function MDXSection({ children }: MDXSectionProps) {
  return (
    <div className="flex flex-col h-full w-full justify-center bg-bgLessDark">
      <div className="flex w-9/12 2xl:w-8/12 flex-col rounded-sm bg-white p-16 mx-auto my-12 gap-3">
        <MDXProvider components={components}>{children}</MDXProvider>
      </div>
    </div>
  );
}
