import { motion } from "framer-motion";
import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";

export function Legend() {
  return (
    <motion.div
      initial={{ y: 16, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
      className="glass pointer-events-auto absolute bottom-5 left-5 z-20 hidden rounded-2xl px-4 py-3 shadow-glass md:block"
    >
      <div className="text-[10.5px] uppercase tracking-[0.18em] text-white/40">
        Predicted net flow
      </div>
      <div className="mt-2 flex items-center gap-4">
        <Item color="#34e3c2" Icon={ArrowDownRight} label="Inflow" />
        <Item color="#9bb0d3" Icon={Minus} label="Balanced" />
        <Item color="#ff6f91" Icon={ArrowUpRight} label="Outflow" />
      </div>
    </motion.div>
  );
}

function Item({
  color,
  Icon,
  label,
}: {
  color: string;
  Icon: typeof ArrowUpRight;
  label: string;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span
        className="inline-flex h-4 w-4 items-center justify-center rounded-full"
        style={{ backgroundColor: `${color}1f`, boxShadow: `0 0 0 1px ${color}55 inset` }}
      >
        <Icon className="h-2.5 w-2.5" style={{ color }} strokeWidth={2.6} />
      </span>
      <span className="text-[11.5px] text-white/70">{label}</span>
    </div>
  );
}
