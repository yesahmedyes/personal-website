await import("./src/env.js");

import createMDX from "@next/mdx";

/** @type {import("next").NextConfig} */
const config = {
  pageExtensions: ["js", "jsx", "md", "mdx", "ts", "tsx"],
  images: {
    // 250 and 500 make the note cards' srcset candidates match their
    // device-pixel box exactly at 1x/2x. Without them the nearest candidates are
    // 256 and 640, forcing the browser into a non-integer downscale with its
    // low-quality bilinear filter, which visibly softens the line art and undoes
    // the unsharp pass baked into the -500 assets. Rest is the Next.js default.
    imageSizes: [16, 32, 48, 64, 96, 128, 250, 256, 384, 500],
  },
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
