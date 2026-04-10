import { UserButton } from '@clerk/clerk-react';
import { Link, Outlet } from 'react-router-dom';

export default function AppShell() {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="flex items-center justify-between border-b border-gray-200 bg-white px-6 py-3">
        <div className="flex items-center gap-8">
          <Link to="/" className="text-lg font-semibold text-gray-900 tracking-tight">
            Stratos
          </Link>
          <nav className="flex gap-5 text-sm text-gray-500">
            <Link to="/dashboard" className="hover:text-gray-900 transition-colors">
              Dashboard
            </Link>
          </nav>
        </div>
        <UserButton />
      </header>
      <main className="px-6 py-6">
        <Outlet />
      </main>
    </div>
  );
}
