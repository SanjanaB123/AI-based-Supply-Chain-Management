import type { ReactNode } from 'react';

interface Props {
  title: string;
  children: ReactNode;
}

export default function SectionContainer({ title, children }: Props) {
  return (
    <section>
      <h2 className="mb-4 text-xs font-semibold uppercase tracking-wider text-gray-400">
        {title}
      </h2>
      {children}
    </section>
  );
}
