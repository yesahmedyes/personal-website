import Navbar from "./_components/navbar";
import Socials from "./_components/socials";
import NotesList from "./notes/notesList";
import BlogsList from "./blogs/blogsList";

export default function Main() {
  return (
    <div className="flex w-full flex-col">
      <div className="bg-bgDark w-full py-32">
        <div className="flex flex-col lg:flex-row gap-12 lg:gap-10 justify-center place-items-center max-w-4xl mx-auto pb-4">
          <div className="aspect-square cursor-pointer rounded-2xl bg-[#E2D6CD] bg-[url('/images/me.png')] bg-[length:80%] bg-[left_1rem_top_2rem] bg-no-repeat w-2/3 lg:w-1/3"></div>

          <div className="flex flex-col gap-5 lg:gap-4 text-textWhite text-lg px-6 lg:px-0 w-full lg:w-2/3">
            <p className="font-serif font-medium text-xl lg:text-2xl text-center lg:text-left">Hey, I&apos;m Ahmed.</p>
            <p className="font-sans font-light leading-relaxed text-center lg:text-justify pb-0 lg:pb-2">
              An aspiring machine learning researcher with an interest in continual learning, reinforcement learning and large language models.
            </p>
          </div>
        </div>
      </div>

      <Navbar />

      <div className="flex flex-col h-full w-full lg:w-9/12 2xl:w-8/12 place-self-center lg:py-4 px-6 lg:px-0 gap-12">
        <div className="flex flex-col gap-5">
          <div className="text-textBlack text-lg lg:text-xl font-bold text-center lg:text-left">Blogs</div>

          <BlogsList viewAll={false} />
        </div>

        <div className="flex flex-col gap-5">
          <div className="text-textBlack text-lg lg:text-xl font-bold text-center lg:text-left">Notes</div>

          <NotesList />
        </div>
      </div>

      <div className="bg-bgDark w-full py-16 mt-14">
        <div className="flex flex-col flex-grow gap-10 justify-center place-items-center max-w-2xl mx-auto px-6 lg:px-0">
          <p className="text-textLightGray text-sm font-normal leading-relaxed text-center">This website is a personal lookup for all my notes and blogs.</p>

          <Socials />

          <p className="text-textLightGray text-sm font-normal leading-relaxed">Ahmed Haroon © 2025</p>
        </div>
      </div>
    </div>
  );
}
