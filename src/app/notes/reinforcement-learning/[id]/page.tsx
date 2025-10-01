"use client";

import { useEffect, useMemo } from "react";
import { useParams, useRouter } from "next/navigation";
import MDXSection from "../../../_components/markdown/mdxSection";
import { data } from "./data";
import { ArrowLeft2 } from "iconsax-react";

export default function MachineLearning() {
  const { id } = useParams();
  const router = useRouter();

  useEffect(() => {
    if (!id || !data.find((note) => note.id === id)) {
      router.replace("/notes/reinforcement-learning");
    }
  }, [id, router]);

  const title = useMemo(() => data.find((note) => note.id === id)?.title, [id]);
  const Component = useMemo(() => data.find((note) => note.id === id)?.component, [id]);

  return (
    <>
      <ArrowLeft2 size={20} className="fixed top-4 left-4 text-white opacity-50 hover:opacity-100 cursor-pointer" onClick={() => router.replace("/notes/reinforcement-learning")} />
      <MDXSection title={title ?? "Reinforcement Learning"}>{Component && <Component />}</MDXSection>
    </>
  );
}
