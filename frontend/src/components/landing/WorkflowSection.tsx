import { type FC, useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

// ── Icons ──────────────────────────────────────────────────────────────────────

function IconConnect() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
      <circle cx="5" cy="11" r="2.5" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="17" cy="5" r="2.5" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="17" cy="17" r="2.5" stroke="currentColor" strokeWidth="1.5" />
      <path d="M7.5 11h4M12 9.2l5-3.2M12 12.8l5 3.2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function IconAnalyze() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
      <rect x="3" y="3" width="16" height="16" rx="3" stroke="currentColor" strokeWidth="1.5" />
      <path d="M7 15l3-4 2.5 2.5L16 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="16" cy="8" r="1.5" fill="currentColor" />
    </svg>
  );
}

function IconAct() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
      <path d="M11 3l2.5 5.5L19 9.5l-4 3.8 1 5.7L11 16.4 6 19l1-5.7-4-3.8 5.5-1L11 3z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

// ── Connector arrow ────────────────────────────────────────────────────────────

function ConnectorArrow() {
  return (
    <div
      aria-hidden="true"
      className="hidden md:flex md:w-10 md:shrink-0 md:items-center md:justify-center"
    >
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
        <path
          d="M4 10h12M13 6.5l3.5 3.5-3.5 3.5"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="text-slate-300 dark:text-slate-600"
        />
      </svg>
    </div>
  );
}

// ── Data ───────────────────────────────────────────────────────────────────────

interface Step {
  step: string;
  Icon: FC;
  title: string;
  description: string;
}

const STEPS: Step[] = [
  {
    step: '01',
    Icon: IconConnect,
    title: 'Connect your data',
    description:
      'Integrate existing inventory systems, supplier feeds, and sales data through lightweight connectors. No rip-and-replace required.',
  },
  {
    step: '02',
    Icon: IconAnalyze,
    title: 'AI analyzes everything',
    description:
      'Stratos continuously processes 50+ signals per SKU — demand patterns, lead times, seasonal variance, and supplier risk — in real time.',
  },
  {
    step: '03',
    Icon: IconAct,
    title: 'Act on intelligent insights',
    description:
      'Receive precise alerts, demand forecasts, and risk scores before disruptions occur. Your team makes confident moves, not reactive ones.',
  },
];

// ── Section ───────────────────────────────────────────────────────────────────

export default function WorkflowSection() {
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.set('.workflow-step', { opacity: 0, y: 24 });

      ScrollTrigger.create({
        trigger: sectionRef.current,
        start: 'top 78%',
        onEnter: () => {
          gsap.to('.workflow-step', {
            opacity: 1,
            y: 0,
            duration: 0.65,
            stagger: 0.13,
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
      id="workflow"
      className="border-t border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 py-24 md:py-32"
    >
      <div className="mx-auto max-w-480 px-6 lg:px-10">

        {/* Header */}
        <div className="mb-16 text-center">
          <p className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-blue-600 dark:text-blue-400">
            How it works
          </p>
          <h2 className="text-[34px] font-bold tracking-tight text-slate-900 dark:text-slate-100 md:text-[42px]">
            From raw data to confident decisions
          </h2>
          <p className="mx-auto mt-4 max-w-lg text-[16px] leading-relaxed text-slate-500 dark:text-slate-400">
            Three clear steps — from integration to action. No complexity,
            no guesswork.
          </p>
        </div>

        {/* Process row — flex row on desktop, stacked on mobile */}
        <div className="flex flex-col gap-3 md:flex-row md:items-stretch md:gap-0">

          {STEPS.map((s, idx) => (
            <div
              key={s.step}
              className="workflow-step flex flex-1 flex-col md:flex-row md:items-stretch"
            >
              {/* Card */}
              <div
                className="flex flex-1 flex-col rounded-2xl border border-slate-200 dark:border-slate-700/80 bg-white dark:bg-slate-800 p-8"
                style={{ boxShadow: '0 1px 6px 0 rgba(0,0,0,0.05)' }}
              >
                {/* Step badge + inline rule */}
                <div className="mb-6 flex items-center gap-3">
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-900 dark:bg-slate-600 text-[11px] font-bold tracking-tight text-white">
                    {s.step}
                  </span>
                  <div className="h-px flex-1 bg-linear-to-r from-slate-200 dark:from-slate-600 to-transparent" />
                </div>

                {/* Icon */}
                <div className="mb-5 inline-flex h-11 w-11 items-center justify-center rounded-xl bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400">
                  <s.Icon />
                </div>

                {/* Copy */}
                <h3 className="mb-2 text-[16px] font-semibold leading-snug text-slate-900 dark:text-slate-100">
                  {s.title}
                </h3>
                <p className="text-[13.5px] leading-relaxed text-slate-500 dark:text-slate-400">
                  {s.description}
                </p>
              </div>

              {/* Arrow connector — desktop */}
              {idx < STEPS.length - 1 && <ConnectorArrow />}

              {/* Vertical connector — mobile */}
              {idx < STEPS.length - 1 && (
                <div
                  aria-hidden="true"
                  className="flex justify-center md:hidden py-1"
                >
                  <div className="h-4 w-px bg-slate-200 dark:bg-slate-700" />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
