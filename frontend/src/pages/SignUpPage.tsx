import { SignUp } from '@clerk/clerk-react';
import clerkAppearance from '../lib/clerkAppearance';

export default function SignUpPage() {
  return (
    <SignUp
      routing="path"
      path="/sign-up"
      signInUrl="/sign-in"
      fallbackRedirectUrl="/dashboard"
      appearance={clerkAppearance}
    />
  );
}
