import { BrowserRouter, Routes, Route } from 'react-router-dom';
import RootLayout from '../app/RootLayout';
import AuthLayout from '../app/AuthLayout';
import AppShell from '../app/AppShell';
import ProtectedRoute from '../components/ProtectedRoute';
import HomePage from '../pages/HomePage';
import SignInPage from '../pages/SignInPage';
import SignUpPage from '../pages/SignUpPage';
import DashboardPage from '../pages/DashboardPage';

export default function AppRouter() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public — landing */}
        <Route element={<RootLayout />}>
          <Route path="/" element={<HomePage />} />
        </Route>

        {/* Auth — Stratos branding wrapper around Clerk cards */}
        <Route element={<AuthLayout />}>
          <Route path="/sign-in/*" element={<SignInPage />} />
          <Route path="/sign-up/*" element={<SignUpPage />} />
        </Route>

        {/* Protected — requires Clerk session */}
        <Route element={<ProtectedRoute />}>
          <Route element={<AppShell />}>
            <Route path="/dashboard" element={<DashboardPage />} />
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
