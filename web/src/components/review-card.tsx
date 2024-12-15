import { Review } from "@/interfaces/review";
import { Card, CardContent, CardHeader } from "./ui/card";
import { useEffect, useState } from "react";
import { to } from "await-to-js";
import useTagsStore from "@/stores/tags-store";
import { env } from "@/lib/env";

type FetchState = "loading" | "error" | "success";

interface ReviewCardProps {
  review: Review;
}

export default function ReviewCard(props: ReviewCardProps) {
  const { review } = props;
  const [tags, setTags] = useState<string[][]>([]);
  const [fetchState, setFetchState] = useState<FetchState>("loading");
  const { reviewIdToTags, saveTags } = useTagsStore();

  useEffect(() => {
    const fetchTags = async () => {
      const endpoint = `${process.env.NEXT_PUBLIC_API_URL}/tag`;
      const [fetchError, response] = await to(
        fetch(endpoint, {
          method: "POST",
          body: review.text,
        }),
      );

      if (fetchError || !response.ok) {
        setFetchState("error");
        return;
      }

      const [jsonError, tags] = await to(response.json());

      if (jsonError) {
        setFetchState("error");
        return;
      }

      setTags(tags);
      saveTags(review.id, tags);
      setFetchState("success");
    };

    if (tags.length > 0) {
      return
    }

    if (review.id in reviewIdToTags) {
      setTags(reviewIdToTags[review.id]);
      setFetchState("success");
    } else {
      fetchTags();
    }
  }, [review]);

  const getTagStyle = (tag: string) => {
    if (tag.endsWith("POS")) return "bg-green-300 px-2";
    if (tag.endsWith("NEG")) return "bg-red-300 px-2";
    if (tag.endsWith("NEU")) return "bg-yellow-300 px-2";
    return "";
  };

  const renderTags = (tags: string[][]) => {
    return (
      <div
        className="flex flex-wrap text-xl leading-relaxed"
        style={{ lineHeight: "1.5em" }}
      >
        {tags.map(([word, tag], index) => (
          <span
            className={`px-2 my-1 rounded-md font-semibold ${getTagStyle(tag)}`}
            key={index}
          >
            {word}
          </span>
        ))}
      </div>
    );
  }

  const renderCardBody = () => {
    switch (fetchState) {
      case "loading":
        return <div>Analyzing review...</div>;

      case "error":
        return <div>Failed to analyze review.</div>;

      case "success":
        return renderTags(tags)
    }
  };

  return (
    <Card>
      <CardHeader className="text-lg italic">{review.text}</CardHeader>
      <CardContent>{renderCardBody()}</CardContent>
    </Card>
  );
}
