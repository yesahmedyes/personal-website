import LiteratureReviewsList from "./blogsList";
import Navbar from "../_components/navbar";

export default function PaperSummaries() {
  return (
    <div className="flex w-full">
      <div className="flex flex-col h-full w-full lg:w-9/12 2xl:w-8/12 place-self-center lg:py-2 px-6 lg:px-0 gap-12 mx-auto">
        <Navbar />

        <LiteratureReviewsList viewAll={true} />
      </div>
    </div>
  );
}
