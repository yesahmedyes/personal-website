import type { ReactNode } from "react";

interface FigureProps {
  src: string;
  alt: string;
  children?: ReactNode;
  className?: string;
  imageClassName?: string;
}

export default function Figure({ src, alt, children, className = "py-6 my-0", imageClassName = "mx-auto w-full max-w-xl" }: FigureProps) {
  return (
    <figure className={className}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={alt} className={imageClassName} />
      {children ? <figcaption className="mt-1 text-center text-sm text-textBlack/70">{children}</figcaption> : null}
    </figure>
  );
}
