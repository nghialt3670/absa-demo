import { Review } from "@/interfaces/review";
import { create } from "zustand";

interface ReviewsStore {
  placeIdToReviews: Record<string, Review[]>;
  saveReviews: (placeId: string, reviews: Review[]) => void;
}

const useReviewsStore = create<ReviewsStore>((set) => ({
  placeIdToReviews: {},

  saveReviews: (placeId, reviews) =>
    set((state) => ({
      placeIdToReviews: {
        ...state.placeIdToReviews,
        [placeId]: reviews,
      },
    })),
}));

export default useReviewsStore;
