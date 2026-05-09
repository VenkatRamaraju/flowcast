/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter Variable",
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono Variable",
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
      colors: {
        ink: {
          950: "#08090c",
          900: "#0b0d12",
          800: "#10131a",
          700: "#161a23",
          600: "#1c2230",
          500: "#252b3b",
          400: "#3a4256",
        },
        flow: {
          // Cool teal for inflow (positive net flow)
          inflow: "#34e3c2",
          inflowSoft: "#7af0d8",
          // Warm coral for outflow (negative net flow)
          outflow: "#ff6f91",
          outflowSoft: "#ff9bb1",
          neutral: "#8aa1c4",
        },
      },
      boxShadow: {
        "glass": "0 1px 0 rgba(255,255,255,0.04) inset, 0 20px 60px -20px rgba(0,0,0,0.6)",
        "panel": "0 30px 80px -30px rgba(0,0,0,0.65), 0 1px 0 rgba(255,255,255,0.05) inset",
        "marker": "0 0 0 1.5px rgba(255,255,255,0.85), 0 6px 14px -2px rgba(0,0,0,0.55)",
      },
      keyframes: {
        pulseRing: {
          "0%": { transform: "scale(0.6)", opacity: "0.7" },
          "100%": { transform: "scale(2.2)", opacity: "0" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        floatY: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-2px)" },
        },
      },
      animation: {
        pulseRing: "pulseRing 2.2s cubic-bezier(0.16, 1, 0.3, 1) infinite",
        shimmer: "shimmer 2s linear infinite",
        floatY: "floatY 3s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
