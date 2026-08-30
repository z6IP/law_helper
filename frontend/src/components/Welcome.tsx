interface WelcomeProps {
  loading?: boolean
}

export function Welcome({ loading }: WelcomeProps) {
  if (loading) {
    return (
      <div className="welcome-skeleton">
        <div className="welcome-skeleton-title" />
        <div className="welcome-skeleton-sub" />
      </div>
    )
  }

  return (
    <div className="welcome">
      <h1>我是小Z</h1>
      <p className="welcome-sub">你的道路交通安全法智能助手，请随时提问</p>
    </div>
  )
}
