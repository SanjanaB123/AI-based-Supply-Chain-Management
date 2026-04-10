import { Link } from 'react-router-dom';
import { useAuth } from '@clerk/clerk-react';

export default function HomePage() {
  const { isSignedIn } = useAuth();

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-white px-6 text-center">
      <h1 className="text-4xl font-bold tracking-tight text-gray-900">Stratos</h1>
      <p className="mt-3 max-w-md text-base text-gray-500">
        AI-powered supply chain intelligence. Predict demand, reduce disruptions, and
        optimize inventory — in one place.
      </p>
      <div className="mt-8 flex gap-3">
        {isSignedIn ? (
          <Link
            to="/dashboard"
            className="rounded-md bg-gray-900 px-5 py-2.5 text-sm font-medium text-white hover:bg-gray-700"
          >
            Go to Dashboard
          </Link>
        ) : (
          <>
            <Link
              to="/sign-in"
              className="rounded-md border border-gray-300 px-5 py-2.5 text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              Sign In
            </Link>
            <Link
              to="/sign-up"
              className="rounded-md bg-gray-900 px-5 py-2.5 text-sm font-medium text-white hover:bg-gray-700"
            >
              Sign Up
            </Link>
          </>
        )}
      </div>
    </div>
  );
}
