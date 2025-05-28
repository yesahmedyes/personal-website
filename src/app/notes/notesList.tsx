import Link from "next/link";
import Image from "next/image";

const notes_data = [
  {
    title: "Machine Learning",
    description: "Supervised Learning, Unsupervised Learning, Learning Theory, etc.",
    image: "/images/machine-learning.png",
    href: "/notes/machine-learning",
  },
];

export default function NotesList() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4 gap-6">
      {notes_data.map((item, index) => (
        <Link key={index} href={item.href}>
          <div className="flex flex-col group w-full rounded-lg border border-black/10 hover:border-black/20 px-4 py-6">
            <div className="group-hover:text-orange-500 text-center text-textBlack font-semibold">{item.title}</div>
            <div className="flex flex-col rounded h-[250px] place-items-center justify-center">
              <Image src={item.image} alt="Notes" width={250} height={300} />
            </div>
            <div className="whitespace-normal text-sm leading-7 text-textBlack text-center">{item.description}</div>
          </div>
        </Link>
      ))}
    </div>
  );
}
