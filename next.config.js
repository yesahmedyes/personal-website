await import("./src/env.js");

import createMDX from "@next/mdx";

/** @type {import("next").NextConfig} */
const config = {
  pageExtensions: ["js", "jsx", "md", "mdx", "ts", "tsx"],
};

const withMDX = createMDX({
  options: {
    // Plugins are referenced by name (not imported function refs) so they are
    // serializable across Turbopack's Rust/JS boundary. Works with webpack too.
    remarkPlugins: [["remark-math"]],
    rehypePlugins: [["rehype-katex"]],
  },
});

export default withMDX(config);
