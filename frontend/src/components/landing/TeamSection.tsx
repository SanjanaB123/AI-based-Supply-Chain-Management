import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

// ── Data ───────────────────────────────────────────────────────────────────────

interface TeamMember {
  name: string;
  contribution: string;
  initials: string;
  avatarBg: string;
  avatarText: string;
}

const TEAM: TeamMember[] = [
  {
    name: 'Aryan Mehta',
    contribution: 'Frontend · Frontend CI/CD · Designing · Branding',
    initials: 'AM',
    avatarBg: 'bg-red-100',
    avatarText: 'text-red-700',
  },
  {
    name: 'Sanjana Brahmbhatt',
    contribution: 'MLOps · ETL',
    initials: 'SB',
    avatarBg: 'bg-blue-100',
    avatarText: 'text-blue-700',
  },
  {
    name: 'Rohit Prabu',
    contribution: 'MLOps · ETL',
    initials: 'RP',
    avatarBg: 'bg-indigo-100',
    avatarText: 'text-indigo-700',
  },
  {
    name: 'Vedashree Bane',
    contribution: 'Containerization · ML Models CI/CD',
    initials: 'VB',
    avatarBg: 'bg-emerald-100',
    avatarText: 'text-emerald-700',
  },
  {
    name: 'Somya Padhy',
    contribution: 'Backend · Backend Deployment',
    initials: 'SP',
    avatarBg: 'bg-amber-100',
    avatarText: 'text-amber-700',
  },
  {
    name: 'Ghanashyam Vagale',
    contribution: 'Documentation',
    initials: 'GV',
    avatarBg: 'bg-slate-100',
    avatarText: 'text-slate-600',
  },
];

// ── Section ───────────────────────────────────────────────────────────────────

export default function TeamSection() {
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.set('.team-card', { opacity: 0, y: 20 });

      ScrollTrigger.create({
        trigger: sectionRef.current,
        start: 'top 80%',
        onEnter: () => {
          gsap.to('.team-card', {
            opacity: 1,
            y: 0,
            duration: 0.55,
            stagger: 0.08,
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
      id="team"
      className="border-t border-slate-100 bg-white py-24 md:py-32"
    >
      <div className="mx-auto max-w-480 px-6 lg:px-10">

        {/* Header */}
        <div className="mb-14 text-center">
          <p className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-slate-400">
            Project team
          </p>
          <h2 className="text-[34px] font-bold tracking-tight text-slate-900 md:text-[42px]">
            Built by the Stratos team
          </h2>
          <p className="mx-auto mt-4 max-w-md text-[16px] leading-relaxed text-slate-500">
            A Northeastern University MLOps capstone project, Spring 2026.
          </p>
        </div>

        {/* Team grid — 3 columns desktop · 2 columns tablet · 1 column mobile */}
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {TEAM.map((member) => (
            <div
              key={member.name}
              className="team-card flex flex-col items-center rounded-2xl border border-slate-200 bg-white px-6 py-9 text-center"
              style={{ boxShadow: '0 1px 4px 0 rgba(0,0,0,0.04)' }}
            >
              {/* Avatar */}
              <div
                className={[
                  'flex h-14 w-14 items-center justify-center rounded-full text-[15px] font-semibold',
                  member.avatarBg,
                  member.avatarText,
                ].join(' ')}
              >
                {member.initials}
              </div>

              {/* Name */}
              <p className="mt-4 text-[15px] font-semibold text-slate-900">
                {member.name}
              </p>

              {/* Contribution — allowed to wrap */}
              <p className="mt-1.5 text-[12.5px] leading-relaxed text-slate-500">
                {member.contribution}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
