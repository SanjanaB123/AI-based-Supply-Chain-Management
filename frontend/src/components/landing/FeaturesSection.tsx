import { type FC, useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

// ── Icons ──────────────────────────────────────────────────────────────────────

function IconForecast() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path d="M3 14l4-5 3 3 4-6 3 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M3 17h14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function IconInventory() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <rect x="3" y="8" width="14" height="9" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
      <path d="M7 8V6a3 3 0 016 0v2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M10 12v2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function IconRisk() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path d="M10 3L3 15.5h14L10 3z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M10 9v3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="10" cy="13.5" r="0.75" fill="currentColor" />
    </svg>
  );
}

function IconAlert() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path d="M10 3a5 5 0 015 5v3l1.5 2.5H3.5L5 11V8a5 5 0 015-5z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M8 15.5a2 2 0 004 0" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function IconVisibility() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <ellipse cx="10" cy="10" rx="7" ry="4.5" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="10" cy="10" r="2" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function IconAI() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path d="M10 3v2M10 15v2M3 10h2M15 10h2M5.05 5.05l1.41 1.41M13.54 13.54l1.41 1.41M5.05 14.95l1.41-1.41M13.54 6.46l1.41-1.41" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="10" cy="10" r="3" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

// ── Data ───────────────────────────────────────────────────────────────────────

interface Feature {
  Icon: FC;
  title: string;
  description: string;
  accent: string;       // bg class for icon container (light)
  accentDark: string;   // bg class for icon container (dark)
  iconColor: string;
}

const FEATURES: Feature[] = [
  {
    Icon: IconForecast,
    title: 'Predictive Demand Forecasting',
    description:
      'ML models trained on your historical data deliver 94%+ accurate demand signals up to 90 days ahead, letting you act before shelves go empty.',
    accent: 'bg-blue-50',
    accentDark: 'dark:bg-blue-950/60',
    iconColor: 'text-blue-600 dark:text-blue-400',
  },
  {
    Icon: IconInventory,
    title: 'Real-time Inventory Intelligence',
    description:
      'Live stock health dashboards surface critical, low, and healthy SKUs at a glance — across every store or warehouse in your network.',
    accent: 'bg-emerald-50',
    accentDark: 'dark:bg-emerald-950/60',
    iconColor: 'text-emerald-600 dark:text-emerald-400',
  },
  {
    Icon: IconRisk,
    title: 'Supplier Risk Monitoring',
    description:
      'Track lead times, flag supplier anomalies, and model disruption scenarios before they cascade into stock-outs or overstock situations.',
    accent: 'bg-red-50',
    accentDark: 'dark:bg-red-950/60',
    iconColor: 'text-red-600 dark:text-red-400',
  },
  {
    Icon: IconAlert,
    title: 'Proactive Alert Engine',
    description:
      'Configurable thresholds trigger instant alerts at 12ms latency — so your team responds to risk before customers feel the impact.',
    accent: 'bg-amber-50',
    accentDark: 'dark:bg-amber-950/60',
    iconColor: 'text-amber-600 dark:text-amber-400',
  },
  {
    Icon: IconVisibility,
    title: 'End-to-end Visibility',
    description:
      'One unified view across procurement, warehousing, and distribution. No more siloed spreadsheets or delayed status reports.',
    accent: 'bg-indigo-50',
    accentDark: 'dark:bg-indigo-950/60',
    iconColor: 'text-indigo-600 dark:text-indigo-400',
  },
  {
    Icon: IconAI,
    title: 'AI-driven Insights',
    description:
      'Stratos analyzes 50+ data signals per SKU — demand, seasonality, supplier health, variance — and surfaces the insights that matter most.',
    accent: 'bg-slate-100',
    accentDark: 'dark:bg-slate-800',
    iconColor: 'text-slate-600 dark:text-slate-400',
  },
];

// ── Feature card ──────────────────────────────────────────────────────────────

function FeatureCard({ feature }: { feature: Feature }) {
  const cardRef = useRef<HTMLDivElement>(null);

  const handleMouseEnter = () => {
    gsap.to(cardRef.current, {
      y: -5,
      boxShadow: '0 20px 48px -8px rgba(0,0,0,0.11)',
      duration: 0.25,
      ease: 'power2.out',
    });
  };

  const handleMouseLeave = () => {
    gsap.to(cardRef.current, {
      y: 0,
      boxShadow: '0 1px 3px 0 rgba(0,0,0,0.06)',
      duration: 0.3,
      ease: 'power2.inOut',
    });
  };

  return (
    <div
      ref={cardRef}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className="feature-card cursor-default rounded-2xl border border-slate-200 dark:border-slate-700/80 bg-white dark:bg-slate-900 p-7"
      style={{ boxShadow: '0 1px 3px 0 rgba(0,0,0,0.06)' }}
    >
      <div className={[
        'mb-5 inline-flex h-11 w-11 items-center justify-center rounded-xl',
        feature.accent,
        feature.accentDark,
        feature.iconColor,
      ].join(' ')}>
        <feature.Icon />
      </div>
      <h3 className="mb-2.5 text-[15px] font-semibold leading-snug text-slate-900 dark:text-slate-100">
        {feature.title}
      </h3>
      <p className="text-[13.5px] leading-relaxed text-slate-500 dark:text-slate-400">
        {feature.description}
      </p>
    </div>
  );
}

// ── Section ───────────────────────────────────────────────────────────────────

export default function FeaturesSection() {
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.set('.feature-card', { opacity: 0, y: 28 });

      ScrollTrigger.create({
        trigger: sectionRef.current,
        start: 'top 78%',
        onEnter: () => {
          gsap.to('.feature-card', {
            opacity: 1,
            y: 0,
            duration: 0.65,
            stagger: 0.09,
            ease: 'power3.out',
          });
        },
        once: true,
      });
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      id="features"
      className="bg-white dark:bg-slate-950 py-24 md:py-32"
    >
      <div className="mx-auto max-w-480 px-6 lg:px-10">

        {/* Header */}
        <div className="mb-16 text-center">
          <p className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-blue-600 dark:text-blue-400">
            Capabilities
          </p>
          <h2 className="text-[34px] font-bold tracking-tight text-slate-900 dark:text-slate-100 md:text-[42px]">
            Everything your supply chain<br className="hidden md:block" /> needs to stay ahead
          </h2>
          <p className="mx-auto mt-4 max-w-xl text-[16px] leading-relaxed text-slate-500 dark:text-slate-400">
            Stratos integrates intelligence across inventory, forecasting, and risk in a
            single unified platform — no stitching required.
          </p>
        </div>

        {/* Grid */}
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((feature) => (
            <FeatureCard key={feature.title} feature={feature} />
          ))}
        </div>
      </div>
    </section>
  );
}
