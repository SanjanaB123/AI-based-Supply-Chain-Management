import { SignIn } from '@clerk/clerk-react';
import clerkAppearance from '../lib/clerkAppearance';

export default function SignInPage() {
  return (
    <SignIn
      routing="path"
      path="/sign-in"
      signUpUrl="/sign-up"
      fallbackRedirectUrl="/dashboard"
      appearance={clerkAppearance}
    />
  );
}
