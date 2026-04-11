import { useState, useEffect } from 'react';
import { useUser } from '@clerk/clerk-react';
import { useTopBar } from '../app/TopBarContext';
import { useCurrentUser } from '../hooks/useCurrentUser';
import { sendContactMessage } from '../lib/contact';

export default function ContactSupportPage() {
  const { setPageMeta } = useTopBar();
  const { user } = useUser();
  const { getToken } = useCurrentUser();

  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [subject, setSubject] = useState('');
  const [message, setMessage] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [result, setResult] = useState<{ success: boolean; message: string } | null>(null);

  useEffect(() => {
    setPageMeta('Contact Support', 'Get help from the Stratos team');
  }, [setPageMeta]);

  // Pre-fill from Clerk user
  useEffect(() => {
    if (user) {
      setName(user.fullName || user.firstName || '');
      setEmail(user.primaryEmailAddress?.emailAddress || '');
    }
  }, [user]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (isSending) return;

    setIsSending(true);
    setResult(null);

    try {
      const token = await getToken();
      const res = await sendContactMessage({ name, email, subject, message }, token);
      setResult(res);
      if (res.success) {
        setSubject('');
        setMessage('');
      }
    } catch {
      setResult({ success: false, message: 'Failed to send message. Please try again.' });
    } finally {
      setIsSending(false);
    }
  }

  return (
    <div className="flex flex-1 items-start justify-center px-4 py-8 sm:py-12">
      <div className="w-full max-w-lg">
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8 dark:border-slate-700 dark:bg-slate-900">
          {/* Header */}
          <div className="mb-6 text-center">
            <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-blue-50 dark:bg-blue-900/30">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-blue-600 dark:text-blue-400">
                <path d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
              Contact Support
            </h2>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
              Have a question or issue? We'll get back to you as soon as possible.
            </p>
          </div>

          {/* Result banner */}
          {result && (
            <div
              className={`mb-4 rounded-lg px-4 py-3 text-sm ${
                result.success
                  ? 'bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-400'
                  : 'bg-red-50 text-red-600 dark:bg-red-900/20 dark:text-red-400'
              }`}
            >
              {result.message}
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="contact-name" className="mb-1.5 block text-[13px] font-medium text-slate-700 dark:text-slate-300">
                Name
              </label>
              <input
                id="contact-name"
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm text-slate-800 transition-colors focus:border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-100 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200 dark:focus:border-blue-600 dark:focus:ring-blue-900/30"
              />
            </div>

            <div>
              <label htmlFor="contact-email" className="mb-1.5 block text-[13px] font-medium text-slate-700 dark:text-slate-300">
                Email
              </label>
              <input
                id="contact-email"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm text-slate-800 transition-colors focus:border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-100 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200 dark:focus:border-blue-600 dark:focus:ring-blue-900/30"
              />
            </div>

            <div>
              <label htmlFor="contact-subject" className="mb-1.5 block text-[13px] font-medium text-slate-700 dark:text-slate-300">
                Subject
              </label>
              <input
                id="contact-subject"
                type="text"
                required
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                placeholder="Brief description of your issue"
                className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 transition-colors focus:border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-100 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200 dark:placeholder:text-slate-500 dark:focus:border-blue-600 dark:focus:ring-blue-900/30"
              />
            </div>

            <div>
              <label htmlFor="contact-message" className="mb-1.5 block text-[13px] font-medium text-slate-700 dark:text-slate-300">
                Message
              </label>
              <textarea
                id="contact-message"
                required
                rows={5}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Describe your question or issue in detail..."
                className="w-full resize-none rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 transition-colors focus:border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-100 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200 dark:placeholder:text-slate-500 dark:focus:border-blue-600 dark:focus:ring-blue-900/30"
              />
            </div>

            <button
              type="submit"
              disabled={isSending}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
            >
              {isSending ? (
                <>
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  Sending...
                </>
              ) : (
                'Send Message'
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
