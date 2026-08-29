import Link from "next/link";
import { routes } from "@/lib/routes";
import { cn } from "@/lib/utils";

const SIZES = {
  sm: "size-8",
  md: "size-10",
  lg: "size-14",
  hero: "size-40 sm:size-48",
} as const;

export function Logo({
  wordmark = true,
  href = routes.home,
  label = "notate",
  size = "md",
  glow = false,
}: {
  wordmark?: boolean;
  href?: string;
  label?: string;
  size?: keyof typeof SIZES;
  glow?: boolean;
}) {
  return (
    <Link
      href={href}
      className="inline-flex items-center gap-3 text-foreground no-underline hover:opacity-80"
      aria-label={`${label} home`}
    >
      <span className={cn("relative inline-flex shrink-0", SIZES[size])}>
        {glow && (
          <span
            aria-hidden
            className="absolute inset-[-28%] rounded-full bg-[radial-gradient(circle,oklch(0.2_0_0/0.14),transparent_68%)] blur-md"
          />
        )}
        <img
          src="/logo.png?v=7"
          alt=""
          width={size === "hero" ? 192 : 40}
          height={size === "hero" ? 192 : 40}
          className="relative size-full object-contain"
        />
      </span>
      {wordmark && (
        <span className="text-[1.35rem] font-bold leading-none tracking-[-0.06em]">
          {label}
        </span>
      )}
    </Link>
  );
}
