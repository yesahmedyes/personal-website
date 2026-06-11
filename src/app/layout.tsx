import "~/styles/globals.css";
import "aos/dist/aos.css";

import { GeistSans } from "geist/font/sans";
import { type Metadata } from "next";

import "katex/dist/katex.min.css";

export const metadata: Metadata = {
  title: "Ahmed Haroon",
  description: "My Personal Website",
  icons: [{ rel: "icon", url: "/favicon.ico" }],
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${GeistSans.variable} leading-relaxed tracking-wide`}>
      <body>{children}</body>
    </html>
  );
}
