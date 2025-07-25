class Vecto < Formula
  desc "A powerful personal note management system with semantic search capabilities"
  homepage "https://github.com/yswa-var/Vectoria"
  url "https://github.com/yswa-var/Vectoriaarchive/refs/tags/v0.1.0.tar.gz"
  sha256 "SHA256_HERE"
  license "MIT"
  head "https://github.com/yswa-var/Vectoria.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "cargo", "install", *std_cargo_args
  end

  test do
    system "#{bin}/vecto", "--help"
  end
end 