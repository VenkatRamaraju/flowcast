import { useEffect, useRef, useState } from "react";

interface Props {
  value: number;
  duration?: number;
  decimals?: number;
  className?: string;
  signed?: boolean;
}

/**
 * Animates a numeric value with an ease-out curve. Renders tabular digits so
 * the number doesn't jiggle horizontally as it changes.
 */
export function AnimatedNumber({
  value,
  duration = 700,
  decimals = 1,
  className,
  signed = false,
}: Props) {
  const [display, setDisplay] = useState(value);
  const fromRef = useRef(value);
  const startedAt = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    fromRef.current = display;
    startedAt.current = null;

    const step = (ts: number) => {
      if (startedAt.current === null) startedAt.current = ts;
      const t = Math.min(1, (ts - startedAt.current) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const next = fromRef.current + (value - fromRef.current) * eased;
      setDisplay(next);
      if (t < 1) {
        rafRef.current = requestAnimationFrame(step);
      }
    };
    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, duration]);

  const sign = signed ? (display > 0 ? "+" : display < 0 ? "−" : "") : "";
  const magnitude = Math.abs(display).toFixed(decimals);

  return <span className={`tabular ${className ?? ""}`}>{`${sign}${magnitude}`}</span>;
}
