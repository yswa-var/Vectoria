class Vecto < Formula
  desc "A powerful personal note management system with semantic search capabilities"
  homepage "https://github.com/yswa-var/Vectoria"
  url "https://github.com/yswa-var/Vectoriaarchive/refs/tags/v0.1.0.tar.gz"
  sha256 "cc139432d4c815b735e7f6be417dd6b8f0afef5f78eb0b52812d1e3c2832d40d"
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