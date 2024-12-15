import { create } from "zustand";

interface TagsStore {
  reviewIdToTags: Record<string, string[][][]>;
  saveTags: (reviewId: string, tags: string[][][]) => void;
}

const useTagsStore = create<TagsStore>((set) => ({
  reviewIdToTags: {},

  saveTags: (reviewId, tags) =>
    set((state) => ({
      reviewIdToTags: {
        ...state.reviewIdToTags,
        [reviewId]: tags,
      },
    })),
}));

export default useTagsStore;
