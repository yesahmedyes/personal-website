"use client";

import { ArrowLeft2 } from "iconsax-react";
import LinksSection from "../../_components/linksSection";
import { data } from "./[id]/data";
import { useRouter } from "next/navigation";

export default function Page() {
  const router = useRouter();

  return (
    <>
      <ArrowLeft2 size={20} className="fixed top-4 left-4 text-white opacity-50 hover:opacity-100 cursor-pointer" onClick={() => router.replace("/notes")} />

      <div className="flex flex-col h-full w-full justify-center bg-bgLessDark">
        <div className="flex w-9/12 2xl:w-8/12 flex-col rounded-sm bg-white p-16 mx-auto my-12 gap-3">
          <div className="text-textBlack text-xl lg:text-2xl font-semibold text-center pb-4">Machine Learning</div>

          <LinksSection links={data.map((item) => ({ title: item.title, link: `/notes/machine-learning/${item.id}` }))} />

          <div className="text-sm leading-7 pt-4">
            <span className="font-semibold">Note: </span>I wrote these notes for my own reference while I was doing the CS229 Machine Learning course offered by Anand Avati at Stanford University in
            Summer 2020. I hope they are useful for others.
          </div>
        </div>
      </div>
    </>
  );
}
