import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

// ── Data ───────────────────────────────────────────────────────────────────────

interface Metric {
  value: string;
  label: string;
  sub: string;
}

const METRICS: Metric[] = [
  {
    value: '94%',
    label: 'Forecast accuracy',
    sub: 'Across 50+ product categories',
  },
  {
    value: '12ms',
    label: 'Alert latency',
    sub: 'Real-time risk detection',
  },
  {
    value: '3.4×',
    label: 'Faster decisions',
    sub: 'vs. manual inventory analysis',
  },
  {
    value: '50+',
    label: 'Data signals per SKU',
    sub: 'Demand, supplier, and market inputs',
  },
];

// ── Section — intentionally dark in both themes (brand section) ───────────────

export default function MetricsSection() {
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.set('.metric-item', { opacity: 0, y: 20 });

      ScrollTrigger.create({
        trigger: sectionRef.current,
        start: 'top 78%',
        onEnter: () => {
          gsap.to('.metric-item', {
            opacity: 1,
            y: 0,
            duration: 0.65,
            stagger: 0.1,
            ease: 'power3.out',
          });
        },
        once: true,
      });
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <section ref={sectionRef} className="bg-slate-950 py-24 md:py-32">
      <div className="mx-auto max-w-480 px-6 lg:px-10">

        {/* Header */}
        <div className="mb-16 text-center">
          <p className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-blue-400">
            Outcomes
          </p>
          <h2 className="text-[34px] font-bold tracking-tight text-white md:text-[42px]">
            Built for results at scale
          </h2>
          <p className="mx-auto mt-4 max-w-lg text-[16px] leading-relaxed text-slate-400">
            Stratos was engineered around the metrics that matter most to
            supply chain teams operating at speed.
          </p>
        </div>

        {/* Metrics grid */}
        <div className="grid grid-cols-2 gap-x-8 gap-y-14 lg:grid-cols-4">
          {METRICS.map((metric) => (
            <div
              key={metric.label}
              className="metric-item text-center"
            >
              {/* Large value */}
              <div className="text-[52px] font-bold leading-none tracking-tight text-white md:text-[60px] lg:text-[68px]">
                {metric.value}
              </div>

              {/* Label */}
              <div className="mt-3 text-[14px] font-semibold text-blue-400">
                {metric.label}
              </div>

              {/* Sub */}
              <div className="mt-1.5 text-[12px] leading-relaxed text-slate-500">
                {metric.sub}
              </div>
            </div>
          ))}
        </div>

        {/* Divider strip */}
        <div className="mt-20 flex items-center justify-center gap-3">
          <div className="h-px w-24 bg-linear-to-r from-transparent to-slate-700" />
          <div className="h-1 w-1 rounded-full bg-blue-500/60" />
          <div className="h-px w-24 bg-linear-to-l from-transparent to-slate-700" />
        </div>

        {/* Tagline below metrics */}
        <p className="mt-6 text-center text-[13px] text-slate-500">
          Stratos is a course capstone project — figures reflect ML model benchmarks on test data.
        </p>
      </div>
    </section>
  );
}
