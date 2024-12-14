import { MapContext } from "@/contexts/map-context";
import { Place } from "@/interfaces/place";
import mapboxgl from "mapbox-gl";
import { useContext, useEffect, useState } from "react";
import {
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerDescription,
  DrawerFooter,
  DrawerHeader,
  DrawerTitle,
} from "@/components/ui/drawer";
import { Button } from "./ui/button";
import { Review } from "@/interfaces/review";
import ReviewCard from "./review-card";
import Link from "next/link";
import to from "await-to-js";
import { getReviews } from "@/actions/get-reviews";
import useReviewsStore from "@/stores/reviews-store";

type FetchState = "loading" | "error" | "success";

interface PlaceMarkerProps {
  place: Place;
}

export function PlaceMarker(props: PlaceMarkerProps) {
  const { place } = props;

  const map = useContext(MapContext);

  const [marker, setMarker] = useState<mapboxgl.Marker>();
  const [reviews, setReviews] = useState<Review[]>([]);
  const [drawerOpen, setDrawerOpen] = useState<boolean>(false);
  const [fetchState, setFetchState] = useState<FetchState>("loading");

  const { placeIdToReviews, saveReviews } = useReviewsStore();

  useEffect(() => {
    if (!map) {
      return;
    }

    const marker = new mapboxgl.Marker()
      .setLngLat([place.longitude, place.latitude])
      .addTo(map);

    const markerEle = marker.getElement();

    markerEle.classList.add("hover:cursor-pointer");

    markerEle.addEventListener("mouseenter", () => {
      map.getCanvas().style.cursor = "pointer";
    });

    markerEle.addEventListener("click", () => {
      setDrawerOpen(true);
    });

    markerEle.addEventListener("mouseleave", () => {
      map.getCanvas().style.cursor = "";
    });

    setMarker(marker);

    return () => {
      marker.remove();
    };
  }, [map]);

  useEffect(() => {
    const fetchReviews = async () => {
      const [error, reviews] = await to(getReviews(place));

      if (error) {
        setFetchState("error");
        return;
      }

      setReviews(reviews);
      saveReviews(place.id, reviews);
      setFetchState("success");
    };

    if (!drawerOpen) {
      return;
    }

    if (place.id in placeIdToReviews) {
      setReviews(placeIdToReviews[place.id]);
      setFetchState("success");
    } else {
      fetchReviews();
    }
  }, [place, drawerOpen]);

  const renderDrawerBody = () => {
    switch (fetchState) {
      case "loading":
        return (
          <div className="flex justify-center items-center">
            Fetching reviews...
          </div>
        );

      case "error":
        return (
          <div className="flex justify-center items-center">
            Failed to fetch reviews.
          </div>
        );

      case "success":
        return (
          <div className="space-y-4 p-4 overflow-x-hidden overflow-y-scroll">
            {reviews.map((review) => (
              <ReviewCard key={review.id} review={review} />
            ))}
          </div>
        );
    }
  };

  const href =
    place.href ??
    `https://www.google.com/maps/search/?api=1&query=Google&query_place_id=${place.placeId}`;

  return (
    <Drawer open={drawerOpen} onOpenChange={setDrawerOpen}>
      <DrawerContent className="flex flex-col p-0 h-[95%]">
        <DrawerHeader className="mt-0">
          <div className="flex flex-row justify-between items-center">
            <DrawerTitle className="text-3xl">Reviews</DrawerTitle>
            <DrawerClose asChild>
              <Button size="icon" variant="outline">
                X
              </Button>
            </DrawerClose>
          </div>
          <DrawerDescription></DrawerDescription>
        </DrawerHeader>
        {renderDrawerBody()}
        <DrawerFooter className="flex flex-col md:flex-row-reverse">
          <Link href={href} target="_blank">
            <Button className="w-full">Google Maps</Button>
          </Link>
        </DrawerFooter>
      </DrawerContent>
    </Drawer>
  );
}
