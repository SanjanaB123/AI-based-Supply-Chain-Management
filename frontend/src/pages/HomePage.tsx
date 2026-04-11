import LandingNav        from '../components/landing/LandingNav';
import HeroSection       from '../components/landing/HeroSection';
import FeaturesSection   from '../components/landing/FeaturesSection';
import MetricsSection    from '../components/landing/MetricsSection';
import WorkflowSection   from '../components/landing/WorkflowSection';
import TeamSection       from '../components/landing/TeamSection';
import LandingFooter     from '../components/landing/LandingFooter';

export default function HomePage() {
  return (
    <div className="min-h-screen overflow-x-hidden bg-white dark:bg-slate-950">
      <LandingNav />
      <main>
        <HeroSection />
        <FeaturesSection />
        <MetricsSection />
        <WorkflowSection />
      </main>
      <LandingFooter />
    </div>
  );
}
