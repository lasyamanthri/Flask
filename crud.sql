-- phpMyAdmin SQL Dump
-- version 4.9.0.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: May 26, 2021 at 06:47 PM
-- Server version: 10.4.6-MariaDB
-- PHP Version: 7.1.32

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `crud`
--

-- --------------------------------------------------------

--
-- Table structure for table `ip`
--

CREATE TABLE `ip` (
  `id` int(11) NOT NULL,
  `Ip4` int(4) UNSIGNED NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `students`
--

CREATE TABLE `students` (
  `id` int(11) NOT NULL,
  `name` varchar(255) NOT NULL,
  `email` varchar(255) NOT NULL,
  `phone` varchar(255) NOT NULL,
  `Ip4` int(4) UNSIGNED NOT NULL,
  `Count` int(11) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

--
-- Dumping data for table `students`
--

INSERT INTO `students` (`id`, `name`, `email`, `phone`, `Ip4`, `Count`) VALUES
(33, 'Lasya', '17h6a1a05@cvsr.ac.in', '1234567890', 0, 0),
(34, 'Ram ', '17h6a1a05@cvsr.ac.in', '1234567890', 0, 0),
(35, 'Sai Prasad', '17h6a1a05@cvsr.ac.in', '1234567890', 0, 0),
(46, 'Lasya', 'lasyamanthri@gmail.com', '1234567890', 0, 0),
(47, 'Lasya', 'lasyamanthri@gmail.com', '12345', 0, 0),
(48, 'Ram ', '17h6a1a05f0@cvsr.ac.in', '1234567890', 0, 0),
(54, 'Lasya', 'lasyamanthri@gmail.com', '1234567890', 0, 0),
(53, 'Sai Prasad', '17h6a1a05h9@cvsr.ac.in', '12345', 2130706433, 0),
(52, 'Lasya', 'lasyamanthri@gmail.com', '12345', 2130706433, 0),
(55, 'Lasya', 'lasyamanthri@gmail.com', '1234567890', 0, 0),
(56, 'Lasya', 'lasyamanthri@gmail.com', '1234567890', 0, 0),
(57, 'Lasya', 'lasyamanthri@gmail.com', '1234567890', 0, 0);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `ip`
--
ALTER TABLE `ip`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `students`
--
ALTER TABLE `students`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `ip`
--
ALTER TABLE `ip`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=26;

--
-- AUTO_INCREMENT for table `students`
--
ALTER TABLE `students`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=58;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
