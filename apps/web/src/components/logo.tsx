import Link from "next/link";
import { routes } from "@/lib/routes";

export function Logo({
  wordmark = true,
  href = routes.home,
  label = "notate",
}: {
  wordmark?: boolean;
  href?: string;
  label?: string;
}) {
  return (
    <Link
      href={href}
      className="inline-flex items-center gap-2.5 text-foreground no-underline hover:opacity-70"
      aria-label={`${label} home`}
    >
      <img
        src="/logo.png?v=7"
        alt=""
        width={28}
        height={28}
        className="size-7 shrink-0"
      />
      {wordmark && (
        <span className="text-[1.2rem] font-bold leading-none tracking-[-0.06em]">
          {label}
        </span>
      )}
    </Link>
  );
}
