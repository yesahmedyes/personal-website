import LiteratureReviewsList from "./blogsList";
import Navbar from "../_components/navbar";

export default function PaperSummaries() {
  return (
    <div className="flex flex-col h-full w-full lg:w-9/12 2xl:w-8/12 place-self-center pb-20 px-6 lg:px-0 gap-3 overflow-y-auto">
      <Navbar />

      <LiteratureReviewsList viewAll={true} />
    </div>
  );
}
